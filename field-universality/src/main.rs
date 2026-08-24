use num_bigint::BigInt;
use num_integer::Integer as _;
use num_traits::{One as _, Signed as _, Zero as _};
use std::collections::{HashSet, VecDeque};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

type IntegerMatrix = Vec<Vec<i128>>;

#[derive(Debug, Clone)]
struct Config {
    local_path: PathBuf,
    blocks: usize,
    schedule: String,
    samples: usize,
    primes: Vec<u64>,
    seed: u64,
    include_rationals: bool,
    distance_only: bool,
    classify_witnesses: bool,
    support_spectrum: bool,
    max_n: usize,
    max_rank_evals: u64,
    max_codewords: u64,
    max_minors: u64,
    output: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            local_path: Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("examples")
                .join("repetition3.matrix"),
            blocks: 2,
            schedule: "APAP".to_owned(),
            samples: 10,
            primes: vec![3, 5],
            seed: 1,
            include_rationals: true,
            distance_only: false,
            classify_witnesses: false,
            support_spectrum: false,
            max_n: 22,
            max_rank_evals: 50_000_000,
            max_codewords: 100_000_000,
            max_minors: 100_000,
            output: None,
        }
    }
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut permutation: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            permutation.swap(i, j);
        }
        permutation
    }
}

fn usage() -> &'static str {
    "field-universality-experiment\n\
\n\
Computes exact coupled deletion-rank profiles for a binary addition-only\n\
encoder interpreted over F_2, selected odd prime fields, and Q.\n\
\n\
USAGE:\n\
  cargo run --release -- [OPTIONS]\n\
\n\
OPTIONS:\n\
  --local PATH            Binary local generator matrix [default: repetition3]\n\
  --blocks N              Number of direct-sum blocks [default: 2]\n\
  --schedule OPS          Operations after B; A=prefix accumulator, P=random\n\
                          permutation [default: APAP]\n\
  --samples N             Coupled permutation samples [default: 10]\n\
  --primes P,Q,...        Odd prime characteristics [default: 3,5]\n\
  --seed N                Reproducible master seed [default: 1]\n\
  --no-rational           Skip the exact characteristic-zero profile\n\
  --distance-only         Enumerate messages; requires --no-rational\n\
  --classify-witnesses    Classify every odd-field minimum witness\n\
  --classify-losses       Deprecated alias for --classify-witnesses\n\
  --support-spectrum      Record minima by exact message support weight\n\
  --max-n N               Refuse exhaustive profiles above N [default: 22]\n\
  --max-rank-evals N      Work-budget guard [default: 50000000]\n\
  --max-codewords N       Message-enumeration guard [default: 100000000]\n\
  --max-minors N          Maximal-minor budget per loss [default: 100000]\n\
  --output PATH           Write CSV to PATH instead of stdout\n\
  -h, --help              Show this help\n"
}

fn parse_usize(name: &str, value: Option<String>) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("missing value for {name}"))?
        .parse::<usize>()
        .map_err(|_| format!("invalid integer for {name}"))
}

fn parse_u64(name: &str, value: Option<String>) -> Result<u64, String> {
    value
        .ok_or_else(|| format!("missing value for {name}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid integer for {name}"))
}

fn parse_config() -> Result<Option<Config>, String> {
    let mut config = Config::default();
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "-h" | "--help" => return Ok(None),
            "--local" => {
                config.local_path = PathBuf::from(
                    arguments
                        .next()
                        .ok_or_else(|| "missing value for --local".to_owned())?,
                );
            }
            "--blocks" => config.blocks = parse_usize("--blocks", arguments.next())?,
            "--schedule" => {
                config.schedule = arguments
                    .next()
                    .ok_or_else(|| "missing value for --schedule".to_owned())?;
            }
            "--samples" => config.samples = parse_usize("--samples", arguments.next())?,
            "--primes" => {
                let text = arguments
                    .next()
                    .ok_or_else(|| "missing value for --primes".to_owned())?;
                config.primes = if text.is_empty() {
                    Vec::new()
                } else {
                    text.split(',')
                        .map(|part| {
                            part.parse::<u64>()
                                .map_err(|_| format!("invalid prime {part:?}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
            }
            "--seed" => config.seed = parse_u64("--seed", arguments.next())?,
            "--no-rational" => config.include_rationals = false,
            "--distance-only" => config.distance_only = true,
            "--classify-witnesses" | "--classify-losses" => config.classify_witnesses = true,
            "--support-spectrum" => config.support_spectrum = true,
            "--max-n" => config.max_n = parse_usize("--max-n", arguments.next())?,
            "--max-rank-evals" => {
                config.max_rank_evals = parse_u64("--max-rank-evals", arguments.next())?
            }
            "--max-codewords" => {
                config.max_codewords = parse_u64("--max-codewords", arguments.next())?
            }
            "--max-minors" => config.max_minors = parse_u64("--max-minors", arguments.next())?,
            "--output" => {
                config.output = Some(PathBuf::from(
                    arguments
                        .next()
                        .ok_or_else(|| "missing value for --output".to_owned())?,
                ));
            }
            _ => return Err(format!("unknown argument {argument:?}")),
        }
    }
    validate_config(&config)?;
    Ok(Some(config))
}

fn validate_config(config: &Config) -> Result<(), String> {
    if config.blocks == 0 {
        return Err("--blocks must be positive".to_owned());
    }
    if config.samples == 0 {
        return Err("--samples must be positive".to_owned());
    }
    if config.distance_only && config.include_rationals {
        return Err("--distance-only requires --no-rational".to_owned());
    }
    if config.support_spectrum && !config.distance_only {
        return Err("--support-spectrum requires --distance-only".to_owned());
    }
    if config.schedule.is_empty()
        || !config
            .schedule
            .bytes()
            .all(|operation| operation == b'A' || operation == b'P')
    {
        return Err("--schedule must be a nonempty string over {A,P}".to_owned());
    }
    for &prime in &config.primes {
        if prime == 2 || !is_prime(prime) {
            return Err(format!("--primes entry {prime} is not an odd prime"));
        }
        if prime > i64::MAX as u64 {
            return Err(format!("prime {prime} exceeds the supported signed range"));
        }
    }
    let mut distinct_primes = HashSet::with_capacity(config.primes.len());
    for &prime in &config.primes {
        if !distinct_primes.insert(prime) {
            return Err(format!("duplicate prime {prime}"));
        }
    }
    Ok(())
}

fn parse_binary_matrix(path: &Path) -> Result<IntegerMatrix, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut matrix = Vec::new();
    let mut width = None;
    for (line_number, raw_line) in contents.lines().enumerate() {
        let line = raw_line.split('#').next().unwrap_or("");
        let bits: Vec<i128> = line
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace() && *byte != b'_')
            .map(|byte| match byte {
                b'0' => Ok(0),
                b'1' => Ok(1),
                _ => Err(format!(
                    "{}:{} contains a character other than 0 or 1",
                    path.display(),
                    line_number + 1
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if bits.is_empty() {
            continue;
        }
        if let Some(expected) = width {
            if bits.len() != expected {
                return Err(format!(
                    "{}:{} has width {}, expected {}",
                    path.display(),
                    line_number + 1,
                    bits.len(),
                    expected
                ));
            }
        } else {
            width = Some(bits.len());
        }
        matrix.push(bits);
    }
    if matrix.is_empty() {
        return Err(format!("{} contains no matrix rows", path.display()));
    }
    if matrix[0].is_empty() {
        return Err(format!("{} has zero width", path.display()));
    }
    Ok(matrix)
}

fn direct_sum(local: &IntegerMatrix, blocks: usize) -> IntegerMatrix {
    let local_rows = local.len();
    let local_columns = local[0].len();
    let mut global = vec![vec![0; local_columns * blocks]; local_rows * blocks];
    for block in 0..blocks {
        for row in 0..local_rows {
            for column in 0..local_columns {
                global[block * local_rows + row][block * local_columns + column] =
                    local[row][column];
            }
        }
    }
    global
}

fn prefix_accumulate(matrix: &mut IntegerMatrix) -> Result<(), String> {
    for row in matrix {
        for column in 1..row.len() {
            row[column] = row[column]
                .checked_add(row[column - 1])
                .ok_or_else(|| "integer overflow while constructing the encoder".to_owned())?;
        }
    }
    Ok(())
}

fn permute_columns(matrix: &mut IntegerMatrix, permutation: &[usize]) {
    for row in matrix {
        let original = row.clone();
        for (output, &input) in permutation.iter().enumerate() {
            row[output] = original[input];
        }
    }
}

fn build_encoder(
    local: &IntegerMatrix,
    blocks: usize,
    schedule: &str,
    seed: u64,
) -> Result<(IntegerMatrix, Vec<Vec<usize>>), String> {
    let mut encoder = direct_sum(local, blocks);
    let length = encoder[0].len();
    let mut random = SplitMix64::new(seed);
    let mut permutations = Vec::new();
    for operation in schedule.bytes() {
        match operation {
            b'A' => prefix_accumulate(&mut encoder)?,
            b'P' => {
                let permutation = random.permutation(length);
                permute_columns(&mut encoder, &permutation);
                permutations.push(permutation);
            }
            _ => unreachable!("schedule was validated"),
        }
    }
    Ok((encoder, permutations))
}

fn sample_seed(master_seed: u64, sample: usize) -> u64 {
    let mut random =
        SplitMix64::new(master_seed ^ (sample as u64).wrapping_mul(0xd6e8_feb8_6659_fd93));
    random.next_u64()
}

fn modular_multiply(left: u64, right: u64, modulus: u64) -> u64 {
    ((left as u128 * right as u128) % modulus as u128) as u64
}

fn modular_power(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1_u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = modular_multiply(result, base, modulus);
        }
        base = modular_multiply(base, base, modulus);
        exponent >>= 1;
    }
    result
}

fn is_prime(candidate: u64) -> bool {
    if candidate < 2 {
        return false;
    }
    for small_prime in [2_u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if candidate == small_prime {
            return true;
        }
        if candidate.is_multiple_of(small_prime) {
            return false;
        }
    }
    let trailing = (candidate - 1).trailing_zeros();
    let odd_part = (candidate - 1) >> trailing;
    // Deterministic Miller--Rabin bases for all 64-bit inputs.
    for base in [2_u64, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022] {
        let reduced = base % candidate;
        if reduced == 0 {
            continue;
        }
        let mut witness = modular_power(reduced, odd_part, candidate);
        if witness == 1 || witness == candidate - 1 {
            continue;
        }
        let mut composite = true;
        for _ in 1..trailing {
            witness = modular_multiply(witness, witness, candidate);
            if witness == candidate - 1 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}

fn integer_mod(value: i128, modulus: u64) -> u64 {
    value.rem_euclid(modulus as i128) as u64
}

struct ModRankWorkspace {
    rows: usize,
    stride: usize,
    data: Vec<u64>,
}

impl ModRankWorkspace {
    fn new(rows: usize, maximum_columns: usize) -> Self {
        Self {
            rows,
            stride: maximum_columns,
            data: vec![0; rows * maximum_columns],
        }
    }

    fn rank(&mut self, matrix: &IntegerMatrix, retained_mask: u64, prime: u64) -> usize {
        let columns = retained_mask.count_ones() as usize;
        if columns == 0 {
            return 0;
        }
        for (row, source) in matrix.iter().enumerate() {
            let mut remaining = retained_mask;
            let mut output_column = 0;
            while remaining != 0 {
                let source_column = remaining.trailing_zeros() as usize;
                self.data[row * self.stride + output_column] =
                    integer_mod(source[source_column], prime);
                output_column += 1;
                remaining &= remaining - 1;
            }
        }

        let mut rank = 0;
        for column in 0..columns {
            let Some(pivot) =
                (rank..self.rows).find(|&row| self.data[row * self.stride + column] != 0)
            else {
                continue;
            };
            if pivot != rank {
                for swap_column in column..columns {
                    self.data.swap(
                        rank * self.stride + swap_column,
                        pivot * self.stride + swap_column,
                    );
                }
            }
            let inverse = modular_power(self.data[rank * self.stride + column], prime - 2, prime);
            for pivot_column in column..columns {
                let index = rank * self.stride + pivot_column;
                self.data[index] = modular_multiply(self.data[index], inverse, prime);
            }
            for row in 0..self.rows {
                if row == rank {
                    continue;
                }
                let factor = self.data[row * self.stride + column];
                if factor == 0 {
                    continue;
                }
                for eliminate_column in column..columns {
                    let index = row * self.stride + eliminate_column;
                    let pivot_entry = self.data[rank * self.stride + eliminate_column];
                    let product = modular_multiply(factor, pivot_entry, prime);
                    let entry = self.data[index];
                    self.data[index] = if entry >= product {
                        entry - product
                    } else {
                        prime - (product - entry)
                    };
                }
            }
            rank += 1;
            if rank == self.rows {
                break;
            }
        }
        rank
    }
}

fn rank_mod_mask(matrix: &IntegerMatrix, retained_mask: u64, prime: u64) -> usize {
    let mut workspace = ModRankWorkspace::new(matrix.len(), matrix[0].len());
    workspace.rank(matrix, retained_mask, prime)
}

fn checked_mul_sub(
    left_a: i128,
    left_b: i128,
    right_a: i128,
    right_b: i128,
) -> Result<i128, String> {
    let left = left_a
        .checked_mul(left_b)
        .ok_or_else(|| "i128 overflow in characteristic-zero rank".to_owned())?;
    let right = right_a
        .checked_mul(right_b)
        .ok_or_else(|| "i128 overflow in characteristic-zero rank".to_owned())?;
    left.checked_sub(right)
        .ok_or_else(|| "i128 overflow in characteristic-zero rank".to_owned())
}

struct RationalRankWorkspace {
    rows: usize,
    stride: usize,
    data: Vec<i128>,
}

impl RationalRankWorkspace {
    fn new(rows: usize, maximum_columns: usize) -> Self {
        Self {
            rows,
            stride: maximum_columns,
            data: vec![0; rows * maximum_columns],
        }
    }

    fn rank(&mut self, matrix: &IntegerMatrix, retained_mask: u64) -> Result<usize, String> {
        let columns = retained_mask.count_ones() as usize;
        if columns == 0 {
            return Ok(0);
        }
        for (row, source) in matrix.iter().enumerate() {
            let mut remaining = retained_mask;
            let mut output_column = 0;
            while remaining != 0 {
                let source_column = remaining.trailing_zeros() as usize;
                self.data[row * self.stride + output_column] = source[source_column];
                output_column += 1;
                remaining &= remaining - 1;
            }
        }

        let mut rank = 0;
        let mut previous_pivot = 1_i128;
        for column in 0..columns {
            let Some(pivot_row) =
                (rank..self.rows).find(|&row| self.data[row * self.stride + column] != 0)
            else {
                continue;
            };
            if pivot_row != rank {
                for swap_column in column..columns {
                    self.data.swap(
                        rank * self.stride + swap_column,
                        pivot_row * self.stride + swap_column,
                    );
                }
            }
            let pivot = self.data[rank * self.stride + column];
            for row in (rank + 1)..self.rows {
                let leading_index = row * self.stride + column;
                let leading = self.data[leading_index];
                for eliminate_column in (column + 1)..columns {
                    let index = row * self.stride + eliminate_column;
                    let pivot_entry = self.data[rank * self.stride + eliminate_column];
                    let numerator = checked_mul_sub(self.data[index], pivot, leading, pivot_entry)?;
                    self.data[index] = if rank == 0 {
                        numerator
                    } else {
                        if numerator % previous_pivot != 0 {
                            return Err("nonexact Bareiss division; internal error".to_owned());
                        }
                        numerator / previous_pivot
                    };
                }
                self.data[leading_index] = 0;
            }
            previous_pivot = pivot;
            rank += 1;
            if rank == self.rows {
                break;
            }
        }
        Ok(rank)
    }
}

#[cfg(test)]
fn rank_rational_mask(matrix: &IntegerMatrix, retained_mask: u64) -> Result<usize, String> {
    let mut workspace = RationalRankWorkspace::new(matrix.len(), matrix[0].len());
    workspace.rank(matrix, retained_mask)
}

fn profile_mod_with_masks(matrix: &IntegerMatrix, prime: u64) -> (Vec<usize>, Vec<u64>) {
    let length = matrix[0].len();
    let subset_count = 1_u64 << length;
    let full_mask = subset_count - 1;
    let mut profile = vec![matrix.len(); length + 1];
    let mut minimizing_masks = vec![u64::MAX; length + 1];
    let mut workspace = ModRankWorkspace::new(matrix.len(), length);
    for deleted_mask in 0..subset_count {
        let deleted = deleted_mask.count_ones() as usize;
        let rank = workspace.rank(matrix, full_mask ^ deleted_mask, prime);
        if minimizing_masks[deleted] == u64::MAX || rank < profile[deleted] {
            profile[deleted] = rank;
            minimizing_masks[deleted] = deleted_mask;
        }
    }
    (profile, minimizing_masks)
}

#[cfg(test)]
fn profile_mod(matrix: &IntegerMatrix, prime: u64) -> Vec<usize> {
    profile_mod_with_masks(matrix, prime).0
}

fn profile_rational(matrix: &IntegerMatrix) -> Result<Vec<usize>, String> {
    let length = matrix[0].len();
    let subset_count = 1_u64 << length;
    let full_mask = subset_count - 1;
    let mut profile = vec![matrix.len(); length + 1];
    let mut workspace = RationalRankWorkspace::new(matrix.len(), length);
    for deleted_mask in 0..subset_count {
        let deleted = deleted_mask.count_ones() as usize;
        let rank = workspace.rank(matrix, full_mask ^ deleted_mask)?;
        profile[deleted] = profile[deleted].min(rank);
    }
    Ok(profile)
}

fn distance_from_profile(profile: &[usize], dimension: usize) -> usize {
    profile
        .iter()
        .position(|&rank| rank < dimension)
        .expect("deleting every column has rank zero")
}

struct ModularWitness {
    deleted_mask: u64,
    message: Vec<u64>,
    codeword: Vec<u64>,
}

fn modular_left_kernel_vector(matrix: &IntegerMatrix, retained_mask: u64, prime: u64) -> Vec<u64> {
    let variables = matrix.len();
    let equations = retained_mask.count_ones() as usize;
    let mut work = vec![0_u64; equations * variables];
    let mut remaining = retained_mask;
    let mut equation = 0;
    while remaining != 0 {
        let column = remaining.trailing_zeros() as usize;
        for variable in 0..variables {
            work[equation * variables + variable] = integer_mod(matrix[variable][column], prime);
        }
        equation += 1;
        remaining &= remaining - 1;
    }

    let mut rank = 0;
    let mut pivot_columns = Vec::with_capacity(variables);
    for variable in 0..variables {
        let Some(pivot) = (rank..equations).find(|&row| work[row * variables + variable] != 0)
        else {
            continue;
        };
        if pivot != rank {
            for column in 0..variables {
                work.swap(rank * variables + column, pivot * variables + column);
            }
        }
        let inverse = modular_power(work[rank * variables + variable], prime - 2, prime);
        for column in variable..variables {
            let index = rank * variables + column;
            work[index] = modular_multiply(work[index], inverse, prime);
        }
        for row in 0..equations {
            if row == rank {
                continue;
            }
            let factor = work[row * variables + variable];
            if factor == 0 {
                continue;
            }
            for column in variable..variables {
                let index = row * variables + column;
                let product = modular_multiply(factor, work[rank * variables + column], prime);
                let entry = work[index];
                work[index] = if entry >= product {
                    entry - product
                } else {
                    prime - (product - entry)
                };
            }
        }
        pivot_columns.push(variable);
        rank += 1;
        if rank == equations {
            break;
        }
    }
    assert!(
        rank < variables,
        "retained columns must have deficient row rank"
    );

    let mut is_pivot = vec![false; variables];
    for &pivot in &pivot_columns {
        is_pivot[pivot] = true;
    }
    let free = (0..variables)
        .find(|&variable| !is_pivot[variable])
        .expect("rank deficiency supplies a free variable");
    let mut message = vec![0_u64; variables];
    message[free] = 1;
    for (row, &pivot) in pivot_columns.iter().enumerate() {
        let coefficient = work[row * variables + free];
        message[pivot] = if coefficient == 0 {
            0
        } else {
            prime - coefficient
        };
    }
    message
}

fn modular_codeword(matrix: &IntegerMatrix, message: &[u64], prime: u64) -> Vec<u64> {
    (0..matrix[0].len())
        .map(|column| {
            matrix
                .iter()
                .zip(message)
                .fold(0_u64, |sum, (row, &coefficient)| {
                    let term =
                        modular_multiply(coefficient, integer_mod(row[column], prime), prime);
                    ((sum as u128 + term as u128) % prime as u128) as u64
                })
        })
        .collect()
}

fn modular_witness(matrix: &IntegerMatrix, prime: u64, deleted_mask: u64) -> ModularWitness {
    let full_mask = (1_u64 << matrix[0].len()) - 1;
    let message = modular_left_kernel_vector(matrix, full_mask ^ deleted_mask, prime);
    let codeword = modular_codeword(matrix, &message, prime);
    debug_assert_eq!(
        codeword.iter().filter(|&&symbol| symbol != 0).count(),
        deleted_mask.count_ones() as usize
    );
    ModularWitness {
        deleted_mask,
        message,
        codeword,
    }
}

fn checked_power(base: u64, exponent: usize) -> Option<u64> {
    (0..exponent).try_fold(1_u64, |value, _| value.checked_mul(base))
}

struct MessageEnumeration {
    distance: usize,
    witness: ModularWitness,
    support_spectrum: Vec<usize>,
}

fn minimum_distance_by_messages<const SUPPORT_SPECTRUM: bool>(
    matrix: &IntegerMatrix,
    prime: u64,
) -> Result<MessageEnumeration, String> {
    let dimension = matrix.len();
    let length = matrix[0].len();
    let codeword_count = checked_power(prime, dimension)
        .ok_or_else(|| format!("{prime}^{dimension} codewords do not fit in u64"))?;
    debug_assert!(prime <= i64::MAX as u64);
    let mut reduced = Vec::with_capacity(dimension * length);
    for row in matrix {
        reduced.extend(row.iter().map(|&entry| integer_mod(entry, prime)));
    }
    let mut message = vec![0_u64; dimension];
    let mut codeword = vec![0_u64; length];
    let mut best_distance = length + 1;
    let mut best_message = Vec::new();
    let mut best_codeword = Vec::new();
    let mut message_weight = 0;
    let mut support_spectrum = if SUPPORT_SPECTRUM {
        vec![length + 1; dimension + 1]
    } else {
        Vec::new()
    };

    for _ in 1..codeword_count {
        let mut digit = 0;
        loop {
            let row = &reduced[(digit * length)..((digit + 1) * length)];
            for (symbol, &term) in codeword.iter_mut().zip(row) {
                let sum = *symbol + term;
                *symbol = if sum >= prime { sum - prime } else { sum };
            }
            message[digit] += 1;
            if SUPPORT_SPECTRUM && message[digit] == 1 {
                message_weight += 1;
            }
            if message[digit] != prime {
                break;
            }
            message[digit] = 0;
            if SUPPORT_SPECTRUM {
                message_weight -= 1;
            }
            digit += 1;
        }

        let threshold = if SUPPORT_SPECTRUM {
            support_spectrum[message_weight]
        } else {
            best_distance
        };
        let mut weight = 0;
        for &symbol in &codeword {
            if symbol != 0 {
                weight += 1;
                if weight >= threshold {
                    break;
                }
            }
        }
        if SUPPORT_SPECTRUM && weight < support_spectrum[message_weight] {
            support_spectrum[message_weight] = weight;
        }
        if weight < best_distance {
            best_distance = weight;
            best_message.clone_from(&message);
            best_codeword.clone_from(&codeword);
            if !SUPPORT_SPECTRUM && best_distance <= 1 {
                break;
            }
        }
    }
    if best_message.is_empty() {
        return Err("message enumeration found no nonzero codeword".to_owned());
    }
    let deleted_mask =
        best_codeword
            .iter()
            .enumerate()
            .fold(0_u64, |mask, (coordinate, &symbol)| {
                if symbol == 0 {
                    mask
                } else {
                    mask | (1_u64 << coordinate)
                }
            });
    Ok(MessageEnumeration {
        distance: best_distance,
        witness: ModularWitness {
            deleted_mask,
            message: best_message,
            codeword: best_codeword,
        },
        support_spectrum,
    })
}

fn enumerate_messages(
    matrix: &IntegerMatrix,
    prime: u64,
    support_spectrum: bool,
) -> Result<MessageEnumeration, String> {
    if support_spectrum {
        minimum_distance_by_messages::<true>(matrix, prime)
    } else {
        minimum_distance_by_messages::<false>(matrix, prime)
    }
}

fn retained_coordinates(length: usize, deleted_mask: u64) -> Vec<usize> {
    (0..length)
        .filter(|&coordinate| deleted_mask & (1_u64 << coordinate) == 0)
        .collect()
}

fn independent_columns_mod_two(
    matrix: &IntegerMatrix,
    retained: &[usize],
) -> Result<Vec<usize>, String> {
    let rows = matrix.len();
    let columns = retained.len();
    let mut work = vec![0_u8; rows * columns];
    for row in 0..rows {
        for (column, &source_column) in retained.iter().enumerate() {
            work[row * columns + column] = integer_mod(matrix[row][source_column], 2) as u8;
        }
    }

    let mut rank = 0;
    let mut pivots = Vec::with_capacity(rows);
    for column in 0..columns {
        let Some(pivot) = (rank..rows).find(|&row| work[row * columns + column] != 0) else {
            continue;
        };
        if pivot != rank {
            for swap_column in column..columns {
                work.swap(rank * columns + swap_column, pivot * columns + swap_column);
            }
        }
        for row in 0..rows {
            if row == rank || work[row * columns + column] == 0 {
                continue;
            }
            for eliminate_column in column..columns {
                work[row * columns + eliminate_column] ^= work[rank * columns + eliminate_column];
            }
        }
        pivots.push(retained[column]);
        rank += 1;
        if rank == rows {
            return Ok(pivots);
        }
    }
    Err(format!(
        "retained matrix has binary rank {rank}, expected {rows}"
    ))
}

struct RationalColumnBasis {
    rank: usize,
    pivot_columns: Vec<usize>,
}

fn independent_columns_rational(
    matrix: &IntegerMatrix,
    retained: &[usize],
) -> Result<RationalColumnBasis, String> {
    let rows = matrix.len();
    let columns = retained.len();
    let mut work = Vec::with_capacity(rows * columns);
    for row in matrix {
        for &column in retained {
            work.push(BigInt::from(row[column]));
        }
    }

    let mut rank = 0;
    let mut previous_pivot = BigInt::one();
    let mut pivot_columns = Vec::with_capacity(rows);
    for column in 0..columns {
        let Some(pivot_row) = (rank..rows).find(|&row| !work[row * columns + column].is_zero())
        else {
            continue;
        };
        if pivot_row != rank {
            for swap_column in column..columns {
                work.swap(
                    rank * columns + swap_column,
                    pivot_row * columns + swap_column,
                );
            }
        }

        let pivot = work[rank * columns + column].clone();
        for row in (rank + 1)..rows {
            let leading = work[row * columns + column].clone();
            for eliminate_column in (column + 1)..columns {
                let numerator = &work[row * columns + eliminate_column] * &pivot
                    - &leading * &work[rank * columns + eliminate_column];
                if rank == 0 {
                    work[row * columns + eliminate_column] = numerator;
                } else {
                    let (quotient, remainder) = numerator.div_rem(&previous_pivot);
                    if !remainder.is_zero() {
                        return Err("nonexact BigInt Bareiss division; internal error".to_owned());
                    }
                    work[row * columns + eliminate_column] = quotient;
                }
            }
            work[row * columns + column] = BigInt::zero();
        }
        previous_pivot = pivot;
        pivot_columns.push(retained[column]);
        rank += 1;
        if rank == rows {
            break;
        }
    }
    Ok(RationalColumnBasis {
        rank,
        pivot_columns,
    })
}

fn bareiss_determinant(
    matrix: &IntegerMatrix,
    selected_columns: &[usize],
) -> Result<BigInt, String> {
    let dimension = matrix.len();
    if selected_columns.len() != dimension {
        return Err("determinant requires one selected column per row".to_owned());
    }
    if dimension == 0 {
        return Ok(BigInt::one());
    }
    let mut work = Vec::with_capacity(dimension * dimension);
    for row in matrix {
        for &column in selected_columns {
            work.push(BigInt::from(row[column]));
        }
    }

    let mut sign = BigInt::one();
    let mut previous_pivot = BigInt::one();
    for pivot_index in 0..dimension {
        let Some(pivot_row) =
            (pivot_index..dimension).find(|&row| !work[row * dimension + pivot_index].is_zero())
        else {
            return Ok(BigInt::zero());
        };
        if pivot_row != pivot_index {
            for column in 0..dimension {
                work.swap(
                    pivot_index * dimension + column,
                    pivot_row * dimension + column,
                );
            }
            sign = -sign;
        }
        if pivot_index + 1 == dimension {
            return Ok(sign * &work[pivot_index * dimension + pivot_index]);
        }

        let pivot = work[pivot_index * dimension + pivot_index].clone();
        for row in (pivot_index + 1)..dimension {
            let leading = work[row * dimension + pivot_index].clone();
            for column in (pivot_index + 1)..dimension {
                let numerator = &work[row * dimension + column] * &pivot
                    - &leading * &work[pivot_index * dimension + column];
                if pivot_index == 0 {
                    work[row * dimension + column] = numerator;
                } else {
                    let (quotient, remainder) = numerator.div_rem(&previous_pivot);
                    if !remainder.is_zero() {
                        return Err("nonexact Bareiss division; internal error".to_owned());
                    }
                    work[row * dimension + column] = quotient;
                }
            }
            work[row * dimension + pivot_index] = BigInt::zero();
        }
        previous_pivot = pivot;
    }
    unreachable!("nonempty determinant loop returns at its final pivot")
}

fn binomial(n: usize, k: usize) -> u64 {
    let k = k.min(n - k);
    let mut value = 1_u128;
    for index in 1..=k {
        value = value * (n - k + index) as u128 / index as u128;
    }
    u64::try_from(value).expect("n < 63 keeps binomial coefficients in u64")
}

fn top_determinantal_divisor(
    matrix: &IntegerMatrix,
    retained: &[usize],
    max_minors: u64,
) -> Result<(u64, Option<BigInt>), String> {
    let dimension = matrix.len();
    if retained.len() < dimension {
        return Ok((0, Some(BigInt::zero())));
    }
    let minor_count = binomial(retained.len(), dimension);
    if minor_count > max_minors {
        return Ok((minor_count, None));
    }

    let mut indices: Vec<usize> = (0..dimension).collect();
    let mut columns = vec![0_usize; dimension];
    let mut divisor = BigInt::zero();
    loop {
        for index in 0..dimension {
            columns[index] = retained[indices[index]];
        }
        divisor = divisor.gcd(&bareiss_determinant(matrix, &columns)?.abs());
        if divisor.is_one() {
            return Ok((minor_count, Some(divisor)));
        }

        let mut position = dimension;
        let mut advanced = false;
        while position != 0 {
            position -= 1;
            let maximum = retained.len() - dimension + position;
            if indices[position] < maximum {
                indices[position] += 1;
                for next in (position + 1)..dimension {
                    indices[next] = indices[next - 1] + 1;
                }
                advanced = true;
                break;
            }
        }
        if !advanced {
            break;
        }
    }
    Ok((minor_count, Some(divisor)))
}

fn prime_valuation(value: &BigInt, prime: u64) -> usize {
    let prime = BigInt::from(prime);
    let mut quotient = value.abs();
    let mut valuation = 0;
    while !quotient.is_zero() && (&quotient % &prime).is_zero() {
        quotient /= &prime;
        valuation += 1;
    }
    valuation
}

struct UnitEquationCore {
    peeled_variables: usize,
    variables: usize,
    message_variables: usize,
    equations: usize,
    edges: usize,
    isolated_variables: usize,
    isolated_message_variables: usize,
    components: usize,
    message_components: usize,
    cycle_rank: usize,
    largest_component_variables: usize,
    largest_component_message_variables: usize,
    largest_component_equations: usize,
    wl_fingerprint: String,
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn hash_color(seed: u128, neighbors: &mut [u128]) -> u128 {
    neighbors.sort_unstable();
    let mut left = mix64(seed as u64 ^ 0x243f_6a88_85a3_08d3);
    let mut right = mix64((seed >> 64) as u64 ^ 0x1319_8a2e_0370_7344);
    for &neighbor in neighbors.iter() {
        left = mix64(left ^ neighbor as u64);
        right = mix64(right ^ (neighbor >> 64) as u64);
    }
    ((right as u128) << 64) | left as u128
}

fn wl_core_fingerprint(
    variable_colors: &[u64],
    equation_colors: &[u64],
    equation_variables: &[Vec<usize>],
    variable_equations: &[Vec<usize>],
    active_variables: &[bool],
    active_equations: &[bool],
) -> String {
    let mut variable_hashes: Vec<u128> = variable_colors
        .iter()
        .map(|&color| {
            ((mix64(color ^ 0xa409_3822_299f_31d0) as u128) << 64)
                | mix64(color ^ 0x082e_fa98_ec4e_6c89) as u128
        })
        .collect();
    let mut equation_hashes: Vec<u128> = equation_colors
        .iter()
        .map(|&color| {
            ((mix64(color ^ 0x4528_21e6_38d0_1377) as u128) << 64)
                | mix64(color ^ 0xbe54_66cf_34e9_0c6c) as u128
        })
        .collect();
    let mut neighbors = Vec::new();
    for _ in 0..32 {
        let mut next_variables = variable_hashes.clone();
        for variable in 0..variable_hashes.len() {
            if !active_variables[variable] {
                continue;
            }
            neighbors.clear();
            neighbors.extend(
                variable_equations[variable]
                    .iter()
                    .filter(|&&equation| active_equations[equation])
                    .map(|&equation| equation_hashes[equation]),
            );
            next_variables[variable] = hash_color(variable_hashes[variable], &mut neighbors);
        }
        let mut next_equations = equation_hashes.clone();
        for equation in 0..equation_hashes.len() {
            if !active_equations[equation] {
                continue;
            }
            neighbors.clear();
            neighbors.extend(
                equation_variables[equation]
                    .iter()
                    .filter(|&&variable| active_variables[variable])
                    .map(|&variable| variable_hashes[variable]),
            );
            next_equations[equation] = hash_color(equation_hashes[equation], &mut neighbors);
        }
        variable_hashes = next_variables;
        equation_hashes = next_equations;
    }
    let mut residual_colors = Vec::new();
    residual_colors.extend(
        variable_hashes
            .iter()
            .enumerate()
            .filter(|(variable, _)| active_variables[*variable])
            .map(|(_, &color)| color),
    );
    residual_colors.extend(
        equation_hashes
            .iter()
            .enumerate()
            .filter(|(equation, _)| active_equations[*equation])
            .map(|(_, &color)| color),
    );
    let fingerprint = hash_color(
        ((active_variables.iter().filter(|&&active| active).count() as u128) << 64)
            | active_equations.iter().filter(|&&active| active).count() as u128,
        &mut residual_colors,
    );
    format!("{fingerprint:032x}")
}

fn peel_unit_equations(
    variables: usize,
    message_variables: usize,
    equation_variables: &[Vec<usize>],
    colors: Option<(&[u64], &[u64])>,
) -> UnitEquationCore {
    let equation_count = equation_variables.len();
    let mut variable_equations = vec![Vec::new(); variables];
    for (equation, support) in equation_variables.iter().enumerate() {
        for &variable in support {
            variable_equations[variable].push(equation);
        }
    }

    let mut active_variables = vec![true; variables];
    let mut active_equations = vec![true; equation_count];
    let mut degrees: Vec<usize> = equation_variables.iter().map(Vec::len).collect();
    let mut queue = VecDeque::with_capacity(equation_count);
    for (equation, &degree) in degrees.iter().enumerate() {
        if degree <= 1 {
            queue.push_back(equation);
        }
    }

    while let Some(equation) = queue.pop_front() {
        if !active_equations[equation] || degrees[equation] > 1 {
            continue;
        }
        active_equations[equation] = false;
        if degrees[equation] == 0 {
            continue;
        }
        let variable = equation_variables[equation]
            .iter()
            .copied()
            .find(|&candidate| active_variables[candidate])
            .expect("a degree-one equation has one active variable");
        active_variables[variable] = false;
        for &affected in &variable_equations[variable] {
            if active_equations[affected] {
                degrees[affected] -= 1;
                if degrees[affected] <= 1 {
                    queue.push_back(affected);
                }
            }
        }
    }

    let residual_variables = active_variables.iter().filter(|&&active| active).count();
    let residual_message_variables = active_variables[..message_variables]
        .iter()
        .filter(|&&active| active)
        .count();
    let residual_equations = active_equations.iter().filter(|&&active| active).count();
    let residual_edges = active_equations
        .iter()
        .enumerate()
        .filter(|(_, active)| **active)
        .map(|(equation, _)| degrees[equation])
        .sum();
    let isolated_variables = variable_equations
        .iter()
        .enumerate()
        .filter(|(variable, equations)| {
            active_variables[*variable]
                && equations
                    .iter()
                    .all(|&equation| !active_equations[equation])
        })
        .count();
    let isolated_message_variables = variable_equations[..message_variables]
        .iter()
        .enumerate()
        .filter(|(variable, equations)| {
            active_variables[*variable]
                && equations
                    .iter()
                    .all(|&equation| !active_equations[equation])
        })
        .count();
    let mut visited_variables = vec![false; variables];
    let mut visited_equations = vec![false; equation_count];
    let mut components = 0;
    let mut message_components = 0;
    let mut cycle_rank = 0;
    let mut largest_component_variables = 0;
    let mut largest_component_message_variables = 0;
    let mut largest_component_equations = 0;
    for root in 0..variables {
        if !active_variables[root] || visited_variables[root] {
            continue;
        }
        components += 1;
        let mut variable_queue = VecDeque::from([root]);
        let mut equation_queue = VecDeque::new();
        visited_variables[root] = true;
        let mut component_variables = 0;
        let mut component_message_variables = 0;
        let mut component_equations = 0;
        let mut component_edges = 0;
        while !variable_queue.is_empty() || !equation_queue.is_empty() {
            while let Some(variable) = variable_queue.pop_front() {
                component_variables += 1;
                component_message_variables += usize::from(variable < message_variables);
                for &equation in &variable_equations[variable] {
                    if active_equations[equation] && !visited_equations[equation] {
                        visited_equations[equation] = true;
                        equation_queue.push_back(equation);
                    }
                }
            }
            while let Some(equation) = equation_queue.pop_front() {
                component_equations += 1;
                component_edges += degrees[equation];
                for &variable in &equation_variables[equation] {
                    if active_variables[variable] && !visited_variables[variable] {
                        visited_variables[variable] = true;
                        variable_queue.push_back(variable);
                    }
                }
            }
        }
        message_components += usize::from(component_message_variables != 0);
        cycle_rank += component_edges + 1 - component_variables - component_equations;
        largest_component_variables = largest_component_variables.max(component_variables);
        largest_component_message_variables =
            largest_component_message_variables.max(component_message_variables);
        largest_component_equations = largest_component_equations.max(component_equations);
    }
    let wl_fingerprint = colors.map_or_else(String::new, |(variable_colors, equation_colors)| {
        wl_core_fingerprint(
            variable_colors,
            equation_colors,
            equation_variables,
            &variable_equations,
            &active_variables,
            &active_equations,
        )
    });
    UnitEquationCore {
        peeled_variables: variables - residual_variables,
        variables: residual_variables,
        message_variables: residual_message_variables,
        equations: residual_equations,
        edges: residual_edges,
        isolated_variables,
        isolated_message_variables,
        components,
        message_components,
        cycle_rank,
        largest_component_variables,
        largest_component_message_variables,
        largest_component_equations,
        wl_fingerprint,
    }
}

fn unit_equation_core(matrix: &IntegerMatrix, retained: &[usize], prime: u64) -> UnitEquationCore {
    let variables = matrix.len();
    let mut equation_variables = vec![Vec::new(); retained.len()];
    for (equation, &coordinate) in retained.iter().enumerate() {
        for (variable, row) in matrix.iter().enumerate() {
            if integer_mod(row[coordinate], prime) != 0 {
                equation_variables[equation].push(variable);
            }
        }
    }
    peel_unit_equations(variables, variables, &equation_variables, None)
}

struct SparseEncoderGraph {
    variables: usize,
    message_variables: usize,
    final_state_start: usize,
    base_equations: Vec<Vec<usize>>,
    variable_colors: Vec<u64>,
    base_equation_colors: Vec<u64>,
    output_equation_color: u64,
    local: IntegerMatrix,
    blocks: usize,
    schedule: Vec<u8>,
    permutations: Vec<Vec<usize>>,
    length: usize,
}

struct CircuitTrace {
    signature: String,
    layer_weights: Vec<usize>,
    layer_runs: Vec<usize>,
    layer_compositions: Vec<String>,
    final_state: Vec<u64>,
}

impl SparseEncoderGraph {
    fn new(
        local: &IntegerMatrix,
        blocks: usize,
        schedule: &str,
        permutations: &[Vec<usize>],
    ) -> Result<Self, String> {
        let local_rows = local.len();
        let local_columns = local[0].len();
        let message_variables = local_rows * blocks;
        let length = local_columns * blocks;
        let state_layers = schedule.len() + 1;
        let variables = message_variables + state_layers * length;
        let state_start = |layer: usize| message_variables + layer * length;
        let mut base_equations = Vec::with_capacity(length * state_layers);
        let mut base_equation_colors = Vec::with_capacity(length * state_layers);
        let mut variable_colors = vec![0_u64; variables];
        for block in 0..blocks {
            for local_row in 0..local_rows {
                variable_colors[block * local_rows + local_row] = 0x1000 + local_row as u64;
            }
        }
        for layer in 0..state_layers {
            for coordinate in 0..length {
                variable_colors[state_start(layer) + coordinate] = 0x2000 + layer as u64;
            }
        }
        let column_patterns: Vec<Vec<i128>> = (0..local_columns)
            .map(|column| local.iter().map(|row| row[column]).collect())
            .collect();
        let mut unique_patterns = column_patterns.clone();
        unique_patterns.sort_unstable();
        unique_patterns.dedup();

        for block in 0..blocks {
            for local_column in 0..local_columns {
                let coordinate = block * local_columns + local_column;
                let mut equation = Vec::with_capacity(local_rows + 1);
                equation.push(state_start(0) + coordinate);
                for (local_row, row) in local.iter().enumerate() {
                    if row[local_column] != 0 {
                        equation.push(block * local_rows + local_row);
                    }
                }
                base_equations.push(equation);
                let pattern_type = unique_patterns
                    .binary_search(&column_patterns[local_column])
                    .expect("local column pattern came from the unique pattern list");
                base_equation_colors.push(0x1_0000 + pattern_type as u64);
            }
        }

        let mut permutation_index = 0;
        for (layer, operation) in schedule.bytes().enumerate() {
            let input = state_start(layer);
            let output = state_start(layer + 1);
            match operation {
                b'A' => {
                    base_equations.push(vec![output, input]);
                    base_equation_colors.push(0x2_0000 + 2 * layer as u64);
                    for coordinate in 1..length {
                        base_equations.push(vec![
                            output + coordinate,
                            output + coordinate - 1,
                            input + coordinate,
                        ]);
                        base_equation_colors.push(0x2_0001 + 2 * layer as u64);
                    }
                }
                b'P' => {
                    let permutation = permutations.get(permutation_index).ok_or_else(|| {
                        "missing permutation while constructing sparse factor graph".to_owned()
                    })?;
                    if permutation.len() != length {
                        return Err("permutation length mismatch in sparse factor graph".to_owned());
                    }
                    for (coordinate, &source) in permutation.iter().enumerate() {
                        base_equations.push(vec![output + coordinate, input + source]);
                        base_equation_colors.push(0x3_0000 + layer as u64);
                    }
                    permutation_index += 1;
                }
                _ => unreachable!("schedule was validated"),
            }
        }
        if permutation_index != permutations.len() {
            return Err("unused permutation in sparse factor graph".to_owned());
        }
        Ok(Self {
            variables,
            message_variables,
            final_state_start: state_start(schedule.len()),
            base_equations,
            variable_colors,
            base_equation_colors,
            output_equation_color: 0x4_0000,
            local: local.clone(),
            blocks,
            schedule: schedule.bytes().collect(),
            permutations: permutations.to_vec(),
            length,
        })
    }

    fn core(&self, retained: &[usize]) -> UnitEquationCore {
        let mut equations = self.base_equations.clone();
        let mut equation_colors = self.base_equation_colors.clone();
        equations.reserve(retained.len());
        equation_colors.reserve(retained.len());
        for &coordinate in retained {
            equations.push(vec![self.final_state_start + coordinate]);
            equation_colors.push(self.output_equation_color);
        }
        peel_unit_equations(
            self.variables,
            self.message_variables,
            &equations,
            Some((&self.variable_colors, &equation_colors)),
        )
    }

    fn trace(&self, message: &[u64], prime: u64) -> Result<CircuitTrace, String> {
        if message.len() != self.message_variables {
            return Err("message length mismatch in sparse circuit trace".to_owned());
        }
        let local_rows = self.local.len();
        let local_columns = self.local[0].len();
        let mut state = vec![0_u64; self.length];
        for block in 0..self.blocks {
            for local_column in 0..local_columns {
                let mut symbol = 0_u64;
                for (local_row, row) in self.local.iter().enumerate() {
                    let term = modular_multiply(
                        message[block * local_rows + local_row],
                        integer_mod(row[local_column], prime),
                        prime,
                    );
                    symbol = ((symbol as u128 + term as u128) % prime as u128) as u64;
                }
                state[block * local_columns + local_column] = symbol;
            }
        }

        let scale = message
            .iter()
            .copied()
            .find(|&symbol| symbol != 0)
            .map(|symbol| modular_power(symbol, prime - 2, prime))
            .ok_or_else(|| "cannot trace the zero message".to_owned())?;
        let normalized_message: Vec<u64> = message
            .iter()
            .map(|&symbol| modular_multiply(symbol, scale, prime))
            .collect();
        let mut states = Vec::with_capacity(self.schedule.len() + 1);
        states.push(state.clone());
        let mut permutation_index = 0;
        for &operation in &self.schedule {
            match operation {
                b'A' => {
                    for coordinate in 1..self.length {
                        state[coordinate] = ((state[coordinate] as u128
                            + state[coordinate - 1] as u128)
                            % prime as u128) as u64;
                    }
                }
                b'P' => {
                    let permutation = &self.permutations[permutation_index];
                    let previous = state.clone();
                    for (coordinate, &source) in permutation.iter().enumerate() {
                        state[coordinate] = previous[source];
                    }
                    permutation_index += 1;
                }
                _ => unreachable!("schedule was validated"),
            }
            states.push(state.clone());
        }

        let layer_weights: Vec<usize> = states
            .iter()
            .map(|layer| layer.iter().filter(|&&symbol| symbol != 0).count())
            .collect();
        let layer_runs: Vec<usize> = states
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .enumerate()
                    .filter(|&(coordinate, symbol)| {
                        *symbol != 0 && (coordinate == 0 || layer[coordinate - 1] == 0)
                    })
                    .count()
            })
            .collect();
        let layer_compositions: Vec<String> = states
            .iter()
            .map(|layer| {
                let normalized: Vec<u64> = layer
                    .iter()
                    .map(|&symbol| modular_multiply(symbol, scale, prime))
                    .collect();
                value_composition(&normalized)
            })
            .collect();
        let mut signature = format!("m={}", value_composition(&normalized_message));
        for layer in 0..states.len() {
            write!(
                signature,
                "|L{layer}:w{}:r{}:{}",
                layer_weights[layer], layer_runs[layer], layer_compositions[layer]
            )
            .expect("writing to a string cannot fail");
        }
        Ok(CircuitTrace {
            signature,
            layer_weights,
            layer_runs,
            layer_compositions,
            final_state: state,
        })
    }
}

struct WitnessDiagnostics {
    class: &'static str,
    message_weight: usize,
    value_composition: String,
    trace_signature: String,
    trace_layer_weights: Vec<usize>,
    trace_layer_runs: Vec<usize>,
    trace_layer_compositions: Vec<String>,
    binary_rank: usize,
    rational_rank: usize,
    smith_2_nullity: usize,
    smith_p_nullity: usize,
    certificate_columns: Vec<usize>,
    certificate_determinant: Option<BigInt>,
    certificate_is_odd: Option<bool>,
    certificate_p_valuation: usize,
    maximal_minor_count: u64,
    top_divisor: Option<BigInt>,
    retained_core: UnitEquationCore,
    factor_core: UnitEquationCore,
}

fn value_composition(message: &[u64]) -> String {
    let mut nonzero: Vec<u64> = message
        .iter()
        .copied()
        .filter(|&symbol| symbol != 0)
        .collect();
    nonzero.sort_unstable();
    let mut output = String::new();
    let mut start = 0;
    while start < nonzero.len() {
        let symbol = nonzero[start];
        let mut end = start + 1;
        while end < nonzero.len() && nonzero[end] == symbol {
            end += 1;
        }
        if !output.is_empty() {
            output.push(';');
        }
        write!(output, "{symbol}:{}", end - start).expect("writing to a string cannot fail");
        start = end;
    }
    output
}

fn diagnose_witness(
    matrix: &IntegerMatrix,
    factor_graph: &SparseEncoderGraph,
    prime: u64,
    distance: usize,
    binary_distance: usize,
    witness: &ModularWitness,
    max_minors: u64,
) -> Result<WitnessDiagnostics, String> {
    let dimension = matrix.len();
    let trace = factor_graph.trace(&witness.message, prime)?;
    if trace.final_state != witness.codeword {
        return Err("sparse circuit trace does not match the generator codeword".to_owned());
    }
    let retained = retained_coordinates(matrix[0].len(), witness.deleted_mask);
    let full_mask = (1_u64 << matrix[0].len()) - 1;
    let retained_mask = full_mask ^ witness.deleted_mask;
    let binary_rank = rank_mod_mask(matrix, retained_mask, 2);
    let prime_rank = rank_mod_mask(matrix, retained_mask, prime);
    if prime_rank == dimension {
        return Err(format!(
            "reported F_{prime} loss witness has full retained rank"
        ));
    }

    let rational_basis = independent_columns_rational(matrix, &retained)?;
    let rational_rank = rational_basis.rank;
    if distance < binary_distance && binary_rank != dimension {
        return Err(format!(
            "witness deletes {distance} coordinates below binary distance {binary_distance}, but retained binary rank is {binary_rank}"
        ));
    }

    let class = if rational_rank != dimension {
        "structural_dependency"
    } else if binary_rank != dimension {
        "shared_modular_dependency"
    } else if distance < binary_distance {
        "distance_beating_p_torsion"
    } else {
        "nonbeating_p_torsion"
    };

    let prime_big = BigInt::from(prime);
    let (
        certificate_columns,
        certificate_determinant,
        certificate_is_odd,
        certificate_p_valuation,
        maximal_minor_count,
        top_divisor,
    ) = if rational_rank == dimension {
        let certificate_columns = if binary_rank == dimension {
            independent_columns_mod_two(matrix, &retained)?
        } else {
            rational_basis.pivot_columns
        };
        let determinant = bareiss_determinant(matrix, &certificate_columns)?;
        if determinant.is_zero() || (&determinant % &prime_big) != BigInt::zero() {
            return Err("invalid prime-torsion certificate; internal error".to_owned());
        }
        let determinant_is_odd = determinant.is_odd();
        if determinant_is_odd != (binary_rank == dimension) {
            return Err("torsion certificate has inconsistent parity; internal error".to_owned());
        }
        let valuation = prime_valuation(&determinant, prime);
        let (minor_count, top_divisor) = top_determinantal_divisor(matrix, &retained, max_minors)?;
        if let Some(divisor) = &top_divisor
            && (divisor.is_zero()
                || (divisor % &prime_big) != BigInt::zero()
                || divisor.is_odd() != (binary_rank == dimension))
        {
            return Err("invalid top determinantal divisor; internal error".to_owned());
        }
        (
            certificate_columns,
            Some(determinant),
            Some(determinant_is_odd),
            valuation,
            minor_count,
            top_divisor,
        )
    } else {
        (
            Vec::new(),
            None,
            None,
            0,
            if retained.len() < dimension {
                0
            } else {
                binomial(retained.len(), dimension)
            },
            Some(BigInt::zero()),
        )
    };

    let retained_core = unit_equation_core(matrix, &retained, prime);
    if retained_core.variables == 0 {
        return Err("rank-deficient system was fully eliminated by unit equations".to_owned());
    }
    let factor_core = factor_graph.core(&retained);
    if factor_core.message_variables == 0 {
        return Err("witness has no message variable in the sparse factor core".to_owned());
    }
    Ok(WitnessDiagnostics {
        class,
        message_weight: witness
            .message
            .iter()
            .filter(|&&symbol| symbol != 0)
            .count(),
        value_composition: value_composition(&witness.message),
        trace_signature: trace.signature,
        trace_layer_weights: trace.layer_weights,
        trace_layer_runs: trace.layer_runs,
        trace_layer_compositions: trace.layer_compositions,
        binary_rank,
        rational_rank,
        smith_2_nullity: dimension - binary_rank,
        smith_p_nullity: dimension - prime_rank,
        certificate_columns,
        certificate_determinant,
        certificate_is_odd,
        certificate_p_valuation,
        maximal_minor_count,
        top_divisor,
        retained_core,
        factor_core,
    })
}

fn profile_text(profile: &[usize]) -> String {
    profile
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(";")
}

fn permutations_text(permutations: &[Vec<usize>]) -> String {
    permutations
        .iter()
        .map(|permutation| {
            permutation
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("-")
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn vector_text(vector: &[u64]) -> String {
    vector
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(";")
}

fn deleted_coordinates_text(mask: u64) -> String {
    (0..64)
        .filter(|&coordinate| mask & (1_u64 << coordinate) != 0)
        .map(|coordinate| coordinate.to_string())
        .collect::<Vec<_>>()
        .join(";")
}

struct CsvSample<'a> {
    sample: usize,
    length: usize,
    dimension: usize,
    schedule: &'a str,
    seed: u64,
    permutations: &'a [Vec<usize>],
}

struct FieldResult<'a> {
    field: &'a str,
    distance: usize,
    profile: &'a [usize],
    support_spectrum: &'a [usize],
    binary_distance: usize,
    witness: Option<&'a ModularWitness>,
    diagnostics: Option<&'a WitnessDiagnostics>,
}

#[derive(Default)]
struct DiagnosticsCsv {
    witness_class: String,
    message_weight: String,
    value_composition: String,
    trace_signature: String,
    trace_layer_weights: String,
    trace_layer_runs: String,
    trace_layer_compositions: String,
    binary_rank: String,
    rational_rank: String,
    smith_2_nullity: String,
    smith_p_nullity: String,
    certificate_columns: String,
    certificate_determinant: String,
    certificate_is_odd: String,
    certificate_p_valuation: String,
    maximal_minor_count: String,
    top_divisor_computed: String,
    top_determinantal_divisor: String,
    retained_peeled_variables: String,
    retained_core_variables: String,
    retained_core_equations: String,
    retained_core_edges: String,
    retained_core_isolated_variables: String,
    factor_peeled_variables: String,
    factor_core_variables: String,
    factor_core_message_variables: String,
    factor_core_equations: String,
    factor_core_edges: String,
    factor_core_isolated_variables: String,
    factor_core_isolated_message_variables: String,
    factor_core_components: String,
    factor_core_message_components: String,
    factor_core_cycle_rank: String,
    factor_core_largest_component_variables: String,
    factor_core_largest_component_message_variables: String,
    factor_core_largest_component_equations: String,
    factor_core_wl_fingerprint: String,
    core_family_signature: String,
}

impl From<&WitnessDiagnostics> for DiagnosticsCsv {
    fn from(diagnostics: &WitnessDiagnostics) -> Self {
        Self {
            witness_class: diagnostics.class.to_owned(),
            message_weight: diagnostics.message_weight.to_string(),
            value_composition: diagnostics.value_composition.clone(),
            trace_signature: diagnostics.trace_signature.clone(),
            trace_layer_weights: diagnostics
                .trace_layer_weights
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(";"),
            trace_layer_runs: diagnostics
                .trace_layer_runs
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(";"),
            trace_layer_compositions: diagnostics.trace_layer_compositions.join("|"),
            binary_rank: diagnostics.binary_rank.to_string(),
            rational_rank: diagnostics.rational_rank.to_string(),
            smith_2_nullity: diagnostics.smith_2_nullity.to_string(),
            smith_p_nullity: diagnostics.smith_p_nullity.to_string(),
            certificate_columns: diagnostics
                .certificate_columns
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(";"),
            certificate_determinant: diagnostics
                .certificate_determinant
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default(),
            certificate_is_odd: diagnostics
                .certificate_is_odd
                .map(|value| value.to_string())
                .unwrap_or_default(),
            certificate_p_valuation: diagnostics.certificate_p_valuation.to_string(),
            maximal_minor_count: diagnostics.maximal_minor_count.to_string(),
            top_divisor_computed: diagnostics.top_divisor.is_some().to_string(),
            top_determinantal_divisor: diagnostics
                .top_divisor
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default(),
            retained_peeled_variables: diagnostics.retained_core.peeled_variables.to_string(),
            retained_core_variables: diagnostics.retained_core.variables.to_string(),
            retained_core_equations: diagnostics.retained_core.equations.to_string(),
            retained_core_edges: diagnostics.retained_core.edges.to_string(),
            retained_core_isolated_variables: diagnostics
                .retained_core
                .isolated_variables
                .to_string(),
            factor_peeled_variables: diagnostics.factor_core.peeled_variables.to_string(),
            factor_core_variables: diagnostics.factor_core.variables.to_string(),
            factor_core_message_variables: diagnostics.factor_core.message_variables.to_string(),
            factor_core_equations: diagnostics.factor_core.equations.to_string(),
            factor_core_edges: diagnostics.factor_core.edges.to_string(),
            factor_core_isolated_variables: diagnostics.factor_core.isolated_variables.to_string(),
            factor_core_isolated_message_variables: diagnostics
                .factor_core
                .isolated_message_variables
                .to_string(),
            factor_core_components: diagnostics.factor_core.components.to_string(),
            factor_core_message_components: diagnostics.factor_core.message_components.to_string(),
            factor_core_cycle_rank: diagnostics.factor_core.cycle_rank.to_string(),
            factor_core_largest_component_variables: diagnostics
                .factor_core
                .largest_component_variables
                .to_string(),
            factor_core_largest_component_message_variables: diagnostics
                .factor_core
                .largest_component_message_variables
                .to_string(),
            factor_core_largest_component_equations: diagnostics
                .factor_core
                .largest_component_equations
                .to_string(),
            factor_core_wl_fingerprint: diagnostics.factor_core.wl_fingerprint.clone(),
            core_family_signature: format!(
                "{}|{}",
                diagnostics.factor_core.wl_fingerprint, diagnostics.trace_signature
            ),
        }
    }
}

fn append_result(csv: &mut String, context: &CsvSample<'_>, result: FieldResult<'_>) {
    let field = result.field;
    let distance = result.distance;
    let binary_distance = result.binary_distance;
    let relative = distance as f64 / context.length as f64;
    let distance_gap = distance as i64 - binary_distance as i64;
    let normalized_gap = distance_gap as f64 / context.length as f64;
    let support_spectrum = result
        .support_spectrum
        .iter()
        .skip(1)
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(";");
    let (deleted_coordinates, message, codeword) = result.witness.map_or_else(
        || (String::new(), String::new(), String::new()),
        |witness| {
            (
                deleted_coordinates_text(witness.deleted_mask),
                vector_text(&witness.message),
                vector_text(&witness.codeword),
            )
        },
    );
    let DiagnosticsCsv {
        witness_class,
        message_weight,
        value_composition,
        trace_signature,
        trace_layer_weights,
        trace_layer_runs,
        trace_layer_compositions,
        binary_rank,
        rational_rank,
        smith_2_nullity,
        smith_p_nullity,
        certificate_columns,
        certificate_determinant,
        certificate_is_odd,
        certificate_p_valuation,
        maximal_minor_count,
        top_divisor_computed,
        top_determinantal_divisor,
        retained_peeled_variables,
        retained_core_variables,
        retained_core_equations,
        retained_core_edges,
        retained_core_isolated_variables,
        factor_peeled_variables,
        factor_core_variables,
        factor_core_message_variables,
        factor_core_equations,
        factor_core_edges,
        factor_core_isolated_variables,
        factor_core_isolated_message_variables,
        factor_core_components,
        factor_core_message_components,
        factor_core_cycle_rank,
        factor_core_largest_component_variables,
        factor_core_largest_component_message_variables,
        factor_core_largest_component_equations,
        factor_core_wl_fingerprint,
        core_family_signature,
    } = result
        .diagnostics
        .map(DiagnosticsCsv::from)
        .unwrap_or_default();
    writeln!(
        csv,
        "{sample},{field},{length},{dimension},{schedule},{seed},{distance},{relative:.12},{binary_distance},{distance_gap},{normalized_gap:.12},\"{profile}\",\"{support_spectrum}\",\"{deleted_coordinates}\",\"{message}\",\"{codeword}\",\"{permutations}\",{witness_class},{message_weight},\"{value_composition}\",\"{trace_signature}\",\"{trace_layer_weights}\",\"{trace_layer_runs}\",\"{trace_layer_compositions}\",{binary_rank},{rational_rank},{smith_2_nullity},{smith_p_nullity},\"{certificate_columns}\",{certificate_determinant},{certificate_is_odd},{certificate_p_valuation},{maximal_minor_count},{top_divisor_computed},{top_determinantal_divisor},{retained_peeled_variables},{retained_core_variables},{retained_core_equations},{retained_core_edges},{retained_core_isolated_variables},{factor_peeled_variables},{factor_core_variables},{factor_core_message_variables},{factor_core_equations},{factor_core_edges},{factor_core_isolated_variables},{factor_core_isolated_message_variables},{factor_core_components},{factor_core_message_components},{factor_core_cycle_rank},{factor_core_largest_component_variables},{factor_core_largest_component_message_variables},{factor_core_largest_component_equations},{factor_core_wl_fingerprint},\"{core_family_signature}\"",
        sample = context.sample,
        length = context.length,
        dimension = context.dimension,
        schedule = context.schedule,
        seed = context.seed,
        profile = profile_text(result.profile),
        permutations = permutations_text(context.permutations),
    )
    .expect("writing to a string cannot fail");
}

fn run(config: &Config) -> Result<String, String> {
    let local = parse_binary_matrix(&config.local_path)?;
    let local_dimension = local.len();
    let local_length = local[0].len();
    if local_length >= 63 {
        return Err("local length must be below 63 for exhaustive subset masks".to_owned());
    }
    if rank_mod_mask(&local, (1_u64 << local_length) - 1, 2) != local_dimension {
        return Err("local binary generator does not have full row rank".to_owned());
    }
    let length = local_length
        .checked_mul(config.blocks)
        .ok_or_else(|| "global length overflow".to_owned())?;
    let dimension = local_dimension
        .checked_mul(config.blocks)
        .ok_or_else(|| "global dimension overflow".to_owned())?;
    if length > config.max_n {
        return Err(format!(
            "global length {length} exceeds --max-n {}",
            config.max_n
        ));
    }
    if length >= 63 {
        return Err("global length must be below 63 for exhaustive subset masks".to_owned());
    }
    if config.distance_only {
        let codewords_per_sample = std::iter::once(2_u64)
            .chain(config.primes.iter().copied())
            .try_fold(0_u64, |total, prime| {
                checked_power(prime, dimension)
                    .and_then(|count| total.checked_add(count.saturating_sub(1)))
            })
            .ok_or_else(|| "codeword-enumeration count overflow".to_owned())?;
        let codewords = codewords_per_sample
            .checked_mul(config.samples as u64)
            .ok_or_else(|| "codeword-enumeration count overflow".to_owned())?;
        if codewords > config.max_codewords {
            return Err(format!(
                "requested {codewords} nonzero codewords, exceeding --max-codewords {}",
                config.max_codewords
            ));
        }
    } else {
        let fields_per_sample =
            1_u64 + config.primes.len() as u64 + u64::from(config.include_rationals);
        let rank_evals = (1_u64 << length)
            .checked_mul(config.samples as u64)
            .and_then(|count| count.checked_mul(fields_per_sample))
            .ok_or_else(|| "rank-evaluation count overflow".to_owned())?;
        if rank_evals > config.max_rank_evals {
            return Err(format!(
                "requested {rank_evals} rank evaluations, exceeding --max-rank-evals {}",
                config.max_rank_evals
            ));
        }
    }

    let mut csv = String::from(
        "sample,field,n,k,schedule,sample_seed,distance,relative_distance,binary_distance,distance_gap,normalized_gap,profile,support_spectrum,deleted_coordinates,message,codeword,permutations,witness_class,message_weight,value_composition,trace_signature,trace_layer_weights,trace_layer_runs,trace_layer_compositions,binary_rank,rational_rank,smith_2_nullity,smith_p_nullity,certificate_columns,certificate_determinant,certificate_is_odd,certificate_p_valuation,maximal_minor_count,top_divisor_computed,top_determinantal_divisor,retained_peeled_variables,retained_core_variables,retained_core_equations,retained_core_edges,retained_core_isolated_variables,factor_peeled_variables,factor_core_variables,factor_core_message_variables,factor_core_equations,factor_core_edges,factor_core_isolated_variables,factor_core_isolated_message_variables,factor_core_components,factor_core_message_components,factor_core_cycle_rank,factor_core_largest_component_variables,factor_core_largest_component_message_variables,factor_core_largest_component_equations,factor_core_wl_fingerprint,core_family_signature\n",
    );
    for sample in 0..config.samples {
        let seed = sample_seed(config.seed, sample);
        let (encoder, permutations) = build_encoder(&local, config.blocks, &config.schedule, seed)?;
        let factor_graph = if config.classify_witnesses {
            Some(SparseEncoderGraph::new(
                &local,
                config.blocks,
                &config.schedule,
                &permutations,
            )?)
        } else {
            None
        };
        let csv_sample = CsvSample {
            sample,
            length,
            dimension,
            schedule: &config.schedule,
            seed,
            permutations: &permutations,
        };
        if config.distance_only {
            let binary = enumerate_messages(&encoder, 2, config.support_spectrum)?;
            let binary_distance = binary.distance;
            append_result(
                &mut csv,
                &csv_sample,
                FieldResult {
                    field: "F2",
                    distance: binary_distance,
                    profile: &[],
                    support_spectrum: &binary.support_spectrum,
                    binary_distance,
                    witness: Some(&binary.witness),
                    diagnostics: None,
                },
            );
            for &prime in &config.primes {
                let enumeration = enumerate_messages(&encoder, prime, config.support_spectrum)?;
                let distance = enumeration.distance;
                let diagnostics = if config.classify_witnesses {
                    Some(diagnose_witness(
                        &encoder,
                        factor_graph
                            .as_ref()
                            .expect("classification constructed a factor graph"),
                        prime,
                        distance,
                        binary_distance,
                        &enumeration.witness,
                        config.max_minors,
                    )?)
                } else {
                    None
                };
                append_result(
                    &mut csv,
                    &csv_sample,
                    FieldResult {
                        field: &format!("F{prime}"),
                        distance,
                        profile: &[],
                        support_spectrum: &enumeration.support_spectrum,
                        binary_distance,
                        witness: Some(&enumeration.witness),
                        diagnostics: diagnostics.as_ref(),
                    },
                );
            }
        } else {
            let (binary_profile, binary_masks) = profile_mod_with_masks(&encoder, 2);
            let binary_distance = distance_from_profile(&binary_profile, dimension);
            let binary_witness = modular_witness(&encoder, 2, binary_masks[binary_distance]);
            append_result(
                &mut csv,
                &csv_sample,
                FieldResult {
                    field: "F2",
                    distance: binary_distance,
                    profile: &binary_profile,
                    support_spectrum: &[],
                    binary_distance,
                    witness: Some(&binary_witness),
                    diagnostics: None,
                },
            );
            for &prime in &config.primes {
                let (profile, masks) = profile_mod_with_masks(&encoder, prime);
                let distance = distance_from_profile(&profile, dimension);
                let witness = modular_witness(&encoder, prime, masks[distance]);
                let diagnostics = if config.classify_witnesses {
                    Some(diagnose_witness(
                        &encoder,
                        factor_graph
                            .as_ref()
                            .expect("classification constructed a factor graph"),
                        prime,
                        distance,
                        binary_distance,
                        &witness,
                        config.max_minors,
                    )?)
                } else {
                    None
                };
                append_result(
                    &mut csv,
                    &csv_sample,
                    FieldResult {
                        field: &format!("F{prime}"),
                        distance,
                        profile: &profile,
                        support_spectrum: &[],
                        binary_distance,
                        witness: Some(&witness),
                        diagnostics: diagnostics.as_ref(),
                    },
                );
            }
            if config.include_rationals {
                let profile = profile_rational(&encoder).map_err(|error| {
                    format!("sample {sample}: {error}; use a smaller instance or --no-rational")
                })?;
                let distance = distance_from_profile(&profile, dimension);
                append_result(
                    &mut csv,
                    &csv_sample,
                    FieldResult {
                        field: "Q",
                        distance,
                        profile: &profile,
                        support_spectrum: &[],
                        binary_distance,
                        witness: None,
                        diagnostics: None,
                    },
                );
            }
        }
    }
    Ok(csv)
}

fn real_main() -> Result<(), String> {
    let Some(config) = parse_config()? else {
        print!("{}", usage());
        return Ok(());
    };
    let csv = run(&config)?;
    if let Some(output) = &config.output {
        if let Some(parent) = output.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }
        fs::write(output, csv)
            .map_err(|error| format!("failed to write {}: {error}", output.display()))?;
    } else {
        print!("{csv}");
    }
    Ok(())
}

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_f2_f3_counterexample() {
        let local = vec![vec![1, 1, 1]];
        let (encoder, permutations) = build_encoder(&local, 1, "AA", 7).unwrap();
        assert!(permutations.is_empty());
        assert_eq!(encoder, vec![vec![1, 3, 6]]);
        assert_eq!(distance_from_profile(&profile_mod(&encoder, 2), 1), 2);
        assert_eq!(distance_from_profile(&profile_mod(&encoder, 3), 1), 1);
        assert_eq!(
            distance_from_profile(&profile_rational(&encoder).unwrap(), 1),
            3
        );
    }

    #[test]
    fn full_space_has_distance_one() {
        let identity = vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
        for prime in [2, 3, 5, 257] {
            assert_eq!(distance_from_profile(&profile_mod(&identity, prime), 3), 1);
        }
        assert_eq!(
            distance_from_profile(&profile_rational(&identity).unwrap(), 3),
            1
        );
    }

    #[test]
    fn rational_and_modular_rank_detect_bad_primes() {
        let even_determinant = vec![vec![1, 1], vec![1, -1]];
        let full_mask = 0b11;
        assert_eq!(rank_rational_mask(&even_determinant, full_mask).unwrap(), 2);
        assert_eq!(rank_mod_mask(&even_determinant, full_mask, 2), 1);
        assert_eq!(rank_mod_mask(&even_determinant, full_mask, 3), 2);

        let determinant_five = vec![vec![1, 1], vec![1, 6]];
        assert_eq!(rank_rational_mask(&determinant_five, full_mask).unwrap(), 2);
        assert_eq!(rank_mod_mask(&determinant_five, full_mask, 5), 1);
    }

    #[test]
    fn modular_witness_realizes_the_minimum_support() {
        let encoder = vec![vec![1, 3, 6]];
        for prime in [2, 3, 5] {
            let (profile, masks) = profile_mod_with_masks(&encoder, prime);
            let distance = distance_from_profile(&profile, 1);
            let witness = modular_witness(&encoder, prime, masks[distance]);
            assert_eq!(
                witness
                    .codeword
                    .iter()
                    .filter(|&&symbol| symbol != 0)
                    .count(),
                distance
            );
            assert!(witness.message.iter().any(|&symbol| symbol != 0));
        }
    }

    #[test]
    fn message_enumeration_matches_rank_profiles() {
        let encoder = vec![vec![1, 3, 6], vec![0, 1, 4]];
        for prime in [2, 3, 5, 7] {
            let profile_distance =
                distance_from_profile(&profile_mod_with_masks(&encoder, prime).0, 2);
            let enumeration = minimum_distance_by_messages::<false>(&encoder, prime).unwrap();
            assert_eq!(enumeration.distance, profile_distance);
            assert_eq!(
                enumeration
                    .witness
                    .codeword
                    .iter()
                    .filter(|&&symbol| symbol != 0)
                    .count(),
                enumeration.distance
            );
        }

        let spectrum_encoder = vec![vec![1, 0, 1], vec![0, 1, 1]];
        let enumeration = minimum_distance_by_messages::<true>(&spectrum_encoder, 3).unwrap();
        assert_eq!(enumeration.distance, 2);
        assert_eq!(enumeration.support_spectrum, vec![4, 2, 2]);
    }

    #[test]
    fn determinant_three_matrix_breaks_field_monotonicity() {
        let generator = vec![
            vec![0, 1, 1, 1, 1],
            vec![1, 0, 1, 1, 1],
            vec![1, 1, 0, 1, 1],
            vec![1, 1, 1, 0, 1],
        ];
        let full_mask = 0b1_1111;
        assert_eq!(rank_mod_mask(&generator, full_mask, 2), 4);
        assert_eq!(rank_mod_mask(&generator, full_mask, 3), 4);
        assert_eq!(
            minimum_distance_by_messages::<false>(&generator, 2)
                .unwrap()
                .distance,
            2
        );
        assert_eq!(
            minimum_distance_by_messages::<false>(&generator, 3)
                .unwrap()
                .distance,
            1
        );
    }

    #[test]
    fn exact_minor_arithmetic_and_top_divisor() {
        let swapped_identity = vec![vec![0, 1], vec![1, 0]];
        assert_eq!(
            bareiss_determinant(&swapped_identity, &[0, 1]).unwrap(),
            BigInt::from(-1)
        );

        let matrix = vec![vec![1, 0, 1], vec![0, 2, 2]];
        let (minor_count, divisor) = top_determinantal_divisor(&matrix, &[0, 1, 2], 3).unwrap();
        assert_eq!(minor_count, 3);
        assert_eq!(divisor, Some(BigInt::from(2)));
        assert_eq!(
            top_determinantal_divisor(&matrix, &[0, 1, 2], 2).unwrap(),
            (3, None)
        );

        let needs_every_minor = vec![vec![6, 10, 15]];
        assert_eq!(
            top_determinantal_divisor(&needs_every_minor, &[0, 1, 2], 3).unwrap(),
            (3, Some(BigInt::one()))
        );
    }

    #[test]
    fn loss_diagnostics_certify_three_torsion() {
        let generator = vec![
            vec![0, 1, 1, 1, 1],
            vec![1, 0, 1, 1, 1],
            vec![1, 1, 0, 1, 1],
            vec![1, 1, 1, 0, 1],
        ];
        let binary = minimum_distance_by_messages::<false>(&generator, 2).unwrap();
        let ternary = minimum_distance_by_messages::<false>(&generator, 3).unwrap();
        let factor_graph = SparseEncoderGraph::new(&generator, 1, "", &[]).unwrap();
        let diagnostics = diagnose_witness(
            &generator,
            &factor_graph,
            3,
            ternary.distance,
            binary.distance,
            &ternary.witness,
            100,
        )
        .unwrap();
        assert_eq!(diagnostics.smith_p_nullity, 1);
        assert_eq!(diagnostics.certificate_columns, vec![0, 1, 2, 3]);
        assert_eq!(diagnostics.certificate_determinant, Some(BigInt::from(-3)));
        assert_eq!(diagnostics.certificate_is_odd, Some(true));
        assert_eq!(diagnostics.certificate_p_valuation, 1);
        assert_eq!(diagnostics.maximal_minor_count, 1);
        assert_eq!(diagnostics.top_divisor, Some(BigInt::from(3)));
        assert_eq!(diagnostics.retained_core.peeled_variables, 0);
        assert_eq!(diagnostics.retained_core.variables, 4);
        assert_eq!(diagnostics.retained_core.equations, 4);
        assert_eq!(diagnostics.retained_core.edges, 12);
        assert_eq!(diagnostics.factor_core.message_variables, 4);
        assert_eq!(diagnostics.class, "distance_beating_p_torsion");
    }

    #[test]
    fn unit_equation_elimination_separates_peelable_and_free_systems() {
        let identity = vec![vec![1, 0], vec![0, 1]];
        let peeled = unit_equation_core(&identity, &[0, 1], 3);
        assert_eq!(peeled.peeled_variables, 2);
        assert_eq!(peeled.variables, 0);
        assert_eq!(peeled.equations, 0);

        let encoder = vec![vec![1, 3, 6]];
        let binary = minimum_distance_by_messages::<false>(&encoder, 2).unwrap();
        let ternary = minimum_distance_by_messages::<false>(&encoder, 3).unwrap();
        let local = vec![vec![1, 1, 1]];
        let factor_graph = SparseEncoderGraph::new(&local, 1, "AA", &[]).unwrap();
        let diagnostics = diagnose_witness(
            &encoder,
            &factor_graph,
            3,
            ternary.distance,
            binary.distance,
            &ternary.witness,
            100,
        )
        .unwrap();
        assert_eq!(diagnostics.top_divisor, Some(BigInt::from(3)));
        assert_eq!(diagnostics.retained_core.variables, 1);
        assert_eq!(diagnostics.retained_core.equations, 0);
        assert_eq!(diagnostics.retained_core.isolated_variables, 1);
        assert!(diagnostics.factor_core.message_variables > 0);
    }

    #[test]
    fn witness_diagnostics_separate_all_algebraic_classes() {
        let local = vec![vec![1, 1, 1]];
        let cases = [
            (
                10_451_216_379_200_822_465,
                "distance_beating_p_torsion",
                2,
                2,
                Some(true),
            ),
            (
                10_858_953_248_184_931_039,
                "nonbeating_p_torsion",
                2,
                2,
                Some(true),
            ),
            (
                11_924_559_415_567_153_392,
                "shared_modular_dependency",
                1,
                2,
                Some(false),
            ),
            (
                3_927_521_153_984_068_587,
                "structural_dependency",
                1,
                1,
                None,
            ),
        ];
        for (seed, expected_class, binary_rank, rational_rank, certificate_is_odd) in cases {
            let (generator, permutations) = build_encoder(&local, 2, "PAPA", seed).unwrap();
            let binary = minimum_distance_by_messages::<false>(&generator, 2).unwrap();
            let ternary = minimum_distance_by_messages::<false>(&generator, 3).unwrap();
            let factor_graph = SparseEncoderGraph::new(&local, 2, "PAPA", &permutations).unwrap();
            let diagnostics = diagnose_witness(
                &generator,
                &factor_graph,
                3,
                ternary.distance,
                binary.distance,
                &ternary.witness,
                100,
            )
            .unwrap();
            assert_eq!(diagnostics.class, expected_class);
            assert_eq!(diagnostics.binary_rank, binary_rank);
            assert_eq!(diagnostics.rational_rank, rational_rank);
            assert_eq!(diagnostics.certificate_is_odd, certificate_is_odd);
        }
    }

    #[test]
    fn primality_check_covers_cli_range() {
        for prime in [2, 3, 5, 257, 65_537, 2_305_843_009_213_693_951] {
            assert!(is_prime(prime));
        }
        for composite in [0, 1, 4, 9, 21, 341, 1_000_000_001] {
            assert!(!is_prime(composite));
        }
    }

    #[test]
    fn deterministic_permutation_samples() {
        let local = vec![vec![1, 1, 1]];
        let first = build_encoder(&local, 2, "APAP", 42).unwrap();
        let second = build_encoder(&local, 2, "APAP", 42).unwrap();
        assert_eq!(first, second);
    }
}

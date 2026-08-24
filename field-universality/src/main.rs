use std::collections::HashSet;
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
    max_n: usize,
    max_rank_evals: u64,
    max_codewords: u64,
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
            max_n: 22,
            max_rank_evals: 50_000_000,
            max_codewords: 100_000_000,
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
  --max-n N               Refuse exhaustive profiles above N [default: 22]\n\
  --max-rank-evals N      Work-budget guard [default: 50000000]\n\
  --max-codewords N       Message-enumeration guard [default: 100000000]\n\
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
            "--max-n" => config.max_n = parse_usize("--max-n", arguments.next())?,
            "--max-rank-evals" => {
                config.max_rank_evals = parse_u64("--max-rank-evals", arguments.next())?
            }
            "--max-codewords" => {
                config.max_codewords = parse_u64("--max-codewords", arguments.next())?
            }
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

fn minimum_distance_by_messages(
    matrix: &IntegerMatrix,
    prime: u64,
) -> Result<(usize, ModularWitness), String> {
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

    for _ in 1..codeword_count {
        let mut digit = 0;
        loop {
            let row = &reduced[(digit * length)..((digit + 1) * length)];
            for (symbol, &term) in codeword.iter_mut().zip(row) {
                let sum = *symbol + term;
                *symbol = if sum >= prime { sum - prime } else { sum };
            }
            message[digit] += 1;
            if message[digit] != prime {
                break;
            }
            message[digit] = 0;
            digit += 1;
        }

        let mut weight = 0;
        for &symbol in &codeword {
            if symbol != 0 {
                weight += 1;
                if weight >= best_distance {
                    break;
                }
            }
        }
        if weight < best_distance {
            best_distance = weight;
            best_message.clone_from(&message);
            best_codeword.clone_from(&codeword);
            if best_distance <= 1 {
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
    Ok((
        best_distance,
        ModularWitness {
            deleted_mask,
            message: best_message,
            codeword: best_codeword,
        },
    ))
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

fn append_result(
    csv: &mut String,
    context: &CsvSample<'_>,
    field: &str,
    distance: usize,
    profile: &[usize],
    binary_distance: usize,
    witness: Option<&ModularWitness>,
) {
    let relative = distance as f64 / context.length as f64;
    let distance_gap = distance as i64 - binary_distance as i64;
    let normalized_gap = distance_gap as f64 / context.length as f64;
    let (deleted_coordinates, message, codeword) = witness.map_or_else(
        || (String::new(), String::new(), String::new()),
        |witness| {
            (
                deleted_coordinates_text(witness.deleted_mask),
                vector_text(&witness.message),
                vector_text(&witness.codeword),
            )
        },
    );
    writeln!(
        csv,
        "{sample},{field},{length},{dimension},{schedule},{seed},{distance},{relative:.12},{binary_distance},{distance_gap},{normalized_gap:.12},\"{profile}\",\"{deleted_coordinates}\",\"{message}\",\"{codeword}\",\"{permutations}\"",
        sample = context.sample,
        length = context.length,
        dimension = context.dimension,
        schedule = context.schedule,
        seed = context.seed,
        profile = profile_text(profile),
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
        "sample,field,n,k,schedule,sample_seed,distance,relative_distance,binary_distance,distance_gap,normalized_gap,profile,deleted_coordinates,message,codeword,permutations\n",
    );
    for sample in 0..config.samples {
        let seed = sample_seed(config.seed, sample);
        let (encoder, permutations) = build_encoder(&local, config.blocks, &config.schedule, seed)?;
        let csv_sample = CsvSample {
            sample,
            length,
            dimension,
            schedule: &config.schedule,
            seed,
            permutations: &permutations,
        };
        if config.distance_only {
            let (binary_distance, binary_witness) = minimum_distance_by_messages(&encoder, 2)?;
            append_result(
                &mut csv,
                &csv_sample,
                "F2",
                binary_distance,
                &[],
                binary_distance,
                Some(&binary_witness),
            );
            for &prime in &config.primes {
                let (distance, witness) = minimum_distance_by_messages(&encoder, prime)?;
                append_result(
                    &mut csv,
                    &csv_sample,
                    &format!("F{prime}"),
                    distance,
                    &[],
                    binary_distance,
                    Some(&witness),
                );
            }
        } else {
            let (binary_profile, binary_masks) = profile_mod_with_masks(&encoder, 2);
            let binary_distance = distance_from_profile(&binary_profile, dimension);
            let binary_witness = modular_witness(&encoder, 2, binary_masks[binary_distance]);
            append_result(
                &mut csv,
                &csv_sample,
                "F2",
                binary_distance,
                &binary_profile,
                binary_distance,
                Some(&binary_witness),
            );
            for &prime in &config.primes {
                let (profile, masks) = profile_mod_with_masks(&encoder, prime);
                let distance = distance_from_profile(&profile, dimension);
                let witness = modular_witness(&encoder, prime, masks[distance]);
                append_result(
                    &mut csv,
                    &csv_sample,
                    &format!("F{prime}"),
                    distance,
                    &profile,
                    binary_distance,
                    Some(&witness),
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
                    "Q",
                    distance,
                    &profile,
                    binary_distance,
                    None,
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
            let (enumerated_distance, witness) =
                minimum_distance_by_messages(&encoder, prime).unwrap();
            assert_eq!(enumerated_distance, profile_distance);
            assert_eq!(
                witness
                    .codeword
                    .iter()
                    .filter(|&&symbol| symbol != 0)
                    .count(),
                enumerated_distance
            );
        }
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
        assert_eq!(minimum_distance_by_messages(&generator, 2).unwrap().0, 2);
        assert_eq!(minimum_distance_by_messages(&generator, 3).unwrap().0, 1);
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

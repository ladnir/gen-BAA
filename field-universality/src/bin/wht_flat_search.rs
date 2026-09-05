use std::collections::{HashMap, HashSet};
use std::env;

const MAX_BASIS: usize = 7;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Flat {
    direction: u128,
    representative: u16,
    color: u16,
}

#[derive(Clone, Copy)]
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
}

fn unpack_basis(packed: u128) -> [u16; MAX_BASIS] {
    let mut basis = [0_u16; MAX_BASIS];
    for (index, row) in basis.iter_mut().enumerate() {
        *row = ((packed >> (16 * index)) & 0xffff) as u16;
    }
    basis
}

fn reduce_basis(mut rows: [u16; MAX_BASIS], n: usize) -> [u16; MAX_BASIS] {
    rows.sort_unstable_by_key(|row| *row == 0);
    let active = rows.iter().take_while(|row| **row != 0).count();
    let mut rank = 0;
    for pivot in (0..n).rev() {
        let Some(selected) = (rank..active).find(|index| rows[*index] & (1_u16 << pivot) != 0)
        else {
            continue;
        };
        rows.swap(rank, selected);
        for other in 0..active {
            if other != rank && rows[other] & (1_u16 << pivot) != 0 {
                rows[other] ^= rows[rank];
            }
        }
        rank += 1;
        if rank == active {
            break;
        }
    }
    rows[rank..].fill(0);
    rows
}

fn pack_basis(rows: [u16; MAX_BASIS]) -> u128 {
    rows.iter()
        .enumerate()
        .fold(0_u128, |packed, (index, row)| {
            packed | ((*row as u128) << (16 * index))
        })
}

fn extend_direction(direction: u128, vector: u16, n: usize) -> u128 {
    let mut rows = unpack_basis(direction);
    let slot = rows
        .iter()
        .position(|row| *row == 0)
        .expect("flat dimension exceeds MAX_BASIS");
    rows[slot] = vector;
    pack_basis(reduce_basis(rows, n))
}

fn canonical_coset_representative(direction: u128, mut point: u16) -> u16 {
    for row in unpack_basis(direction) {
        if row == 0 {
            break;
        }
        let pivot = 15 - row.leading_zeros() as usize;
        if point & (1_u16 << pivot) != 0 {
            point ^= row;
        }
    }
    point
}

fn extend_flats(flats: &[Flat], n: usize) -> Vec<Flat> {
    let mut groups: HashMap<(u128, u16), Vec<u16>> = HashMap::new();
    for flat in flats {
        groups
            .entry((flat.direction, flat.color))
            .or_default()
            .push(flat.representative);
    }

    let mut next = HashSet::new();
    for ((direction, color), representatives) in groups {
        for left in 0..representatives.len() {
            for right in left + 1..representatives.len() {
                let difference = representatives[left] ^ representatives[right];
                let new_direction = extend_direction(direction, difference, n);
                let new_representative =
                    canonical_coset_representative(new_direction, representatives[left]);
                next.insert(Flat {
                    direction: new_direction,
                    representative: new_representative,
                    color,
                });
            }
        }
    }
    next.into_iter().collect()
}

fn flat_points(flat: Flat) -> Vec<usize> {
    let rows = unpack_basis(flat.direction);
    let basis: Vec<u16> = rows.into_iter().take_while(|row| *row != 0).collect();
    let mut points = Vec::with_capacity(1 << basis.len());
    for selector in 0..1_usize << basis.len() {
        let mut point = flat.representative;
        for (index, row) in basis.iter().enumerate() {
            if selector & (1 << index) != 0 {
                point ^= row;
            }
        }
        points.push(point as usize);
    }
    points.sort_unstable();
    points
}

fn fwht_mod(values: &mut [u64], modulus: u64) {
    let mut step = 1;
    while step < values.len() {
        for chunk in values.chunks_exact_mut(2 * step) {
            for index in 0..step {
                let left = chunk[index];
                let right = chunk[index + step];
                chunk[index] = (left + right) % modulus;
                chunk[index + step] = (left + modulus - right) % modulus;
            }
        }
        step *= 2;
    }
}

fn certify_flat_word(
    flat: Flat,
    colors: &[u16],
    blocks: usize,
    alphabet_size: usize,
    modulus: u64,
) -> usize {
    let points = flat_points(flat);
    assert!(points.iter().all(|point| colors[*point] == flat.color));
    let length = colors.len();
    let mut total_weight = 0;

    for block in 0..blocks {
        let multiplier = if block == 0 {
            1
        } else {
            let divisor = alphabet_size.pow((block - 1) as u32);
            ((flat.color as usize / divisor) % alphabet_size + 1) as u64
        };
        let mut spectrum = vec![0_u64; length];
        for point in &points {
            spectrum[*point] = multiplier;
        }
        fwht_mod(&mut spectrum, modulus);
        total_weight += spectrum.iter().filter(|entry| **entry != 0).count();
    }
    total_weight
}

fn parse_arguments() -> Result<(usize, usize, usize, usize, u64), String> {
    let mut n = 10;
    let mut blocks = 4;
    let mut alphabet_size: usize = 2;
    let mut max_dimension = 5;
    let mut seed = 1;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {argument}"))?;
        match argument.as_str() {
            "--n" => n = value.parse().map_err(|_| "invalid --n")?,
            "--blocks" => blocks = value.parse().map_err(|_| "invalid --blocks")?,
            "--alphabet-size" => {
                alphabet_size = value.parse().map_err(|_| "invalid --alphabet-size")?
            }
            "--max-dimension" => {
                max_dimension = value.parse().map_err(|_| "invalid --max-dimension")?
            }
            "--seed" => seed = value.parse().map_err(|_| "invalid --seed")?,
            _ => return Err(format!("unknown argument {argument}")),
        }
    }
    if !(2..=8).contains(&blocks) {
        return Err("--blocks must lie in 2..=8".to_owned());
    }
    if alphabet_size < 2 {
        return Err("--alphabet-size must be at least 2".to_owned());
    }
    if n > 15 {
        return Err("--n must be at most 15".to_owned());
    }
    if max_dimension > MAX_BASIS {
        return Err(format!("--max-dimension must be at most {MAX_BASIS}"));
    }
    let color_count = alphabet_size
        .checked_pow((blocks - 1) as u32)
        .ok_or_else(|| "projective color count overflowed".to_owned())?;
    if color_count > u16::MAX as usize {
        return Err("projective color count must fit in u16".to_owned());
    }
    Ok((n, blocks, alphabet_size, max_dimension, seed))
}

fn main() -> Result<(), String> {
    let (n, blocks, alphabet_size, max_dimension, seed) = parse_arguments()?;
    let length = 1_usize << n;
    let color_count = alphabet_size.pow((blocks - 1) as u32) as u16;
    let mut random = SplitMix64::new(seed);
    let colors: Vec<u16> = (0..length)
        .map(|_| (random.next_u64() % color_count as u64) as u16)
        .collect();
    let mut flats: Vec<Flat> = colors
        .iter()
        .enumerate()
        .map(|(point, color)| Flat {
            direction: 0,
            representative: point as u16,
            color: *color,
        })
        .collect();
    let mut witness = flats[0];
    let mut attained_dimension = 0;

    println!(
        "n={n} length={length} blocks={blocks} alphabet_size={alphabet_size} projective_colors={color_count} seed={seed}"
    );
    println!("dimension=0 monochromatic_flats={}", flats.len());
    for dimension in 1..=max_dimension {
        flats = extend_flats(&flats, n);
        println!("dimension={dimension} monochromatic_flats={}", flats.len());
        if flats.is_empty() {
            break;
        }
        attained_dimension = dimension;
        witness = flats[0];
    }

    let modulus = 65_537;
    let weight = certify_flat_word(witness, &colors, blocks, alphabet_size, modulus);
    let flat_size = 1_usize << attained_dimension;
    let expected_weight = blocks * length / flat_size;
    assert_eq!(weight, expected_weight);
    println!(
        "witness_dimension={attained_dimension} flat_size={flat_size} codeword_weight={weight} relative_weight={:.8}",
        weight as f64 / (blocks * length) as f64
    );
    println!("certificate_expected_weight={expected_weight} modulus={modulus}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerates_all_flats_of_the_three_cube() {
        let mut flats: Vec<Flat> = (0..8)
            .map(|point| Flat {
                direction: 0,
                representative: point,
                color: 0,
            })
            .collect();
        let expected = [28, 14, 1];
        for count in expected {
            flats = extend_flats(&flats, 3);
            assert_eq!(flats.len(), count);
        }
    }

    #[test]
    fn flat_witness_has_the_predicted_weight() {
        let colors = vec![0_u16; 8];
        let mut flats: Vec<Flat> = (0..8)
            .map(|point| Flat {
                direction: 0,
                representative: point,
                color: 0,
            })
            .collect();
        flats = extend_flats(&flats, 3);
        flats = extend_flats(&flats, 3);
        let weight = certify_flat_word(flats[0], &colors, 4, 2, 65_537);
        assert_eq!(weight, 8);
    }
}

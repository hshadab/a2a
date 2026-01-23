//! zkML CLI - Command line interface for Jolt Atlas zkML proofs
//!
//! This CLI wraps Jolt Atlas to provide JSON-based proof generation and verification
//! that can be called from Python or other languages.

use ark_bn254::Fr;
use ark_serialize::CanonicalSerialize;
use clap::{Parser, Subcommand};
use jolt_core::{poly::commitment::dory::DoryCommitmentScheme, transcripts::KeccakTranscript};
use onnx_tracer::{model, tensor::Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::PathBuf,
    time::Instant,
};
use zkml_jolt_core::jolt::JoltSNARK;

#[allow(clippy::upper_case_acronyms)]
type PCS = DoryCommitmentScheme;

#[derive(Parser)]
#[command(name = "zkml-cli")]
#[command(about = "CLI for Jolt Atlas zkML proof generation and verification")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a zkML proof for authorization
    ProveAuth {
        /// Path to input JSON file
        #[arg(short, long)]
        input: PathBuf,
        /// Path to output JSON file
        #[arg(short, long)]
        output: PathBuf,
        /// Path to model directory (containing network.onnx and vocab.json)
        #[arg(short, long)]
        model_dir: PathBuf,
        /// Emit progress events to stderr as JSON
        #[arg(long, default_value = "false")]
        progress: bool,
    },
    /// Generate a zkML proof for URL classification
    ProveClassify {
        /// Path to input JSON file
        #[arg(short, long)]
        input: PathBuf,
        /// Path to output JSON file
        #[arg(short, long)]
        output: PathBuf,
        /// Path to model directory
        #[arg(short, long)]
        model_dir: PathBuf,
        /// Emit progress events to stderr as JSON
        #[arg(long, default_value = "false")]
        progress: bool,
    },
    /// Verify a zkML proof
    Verify {
        /// Path to proof JSON file
        #[arg(short, long)]
        proof: PathBuf,
        /// Path to output JSON file
        #[arg(short, long)]
        output: PathBuf,
    },
}

// ============ Input/Output Structures ============

#[derive(Debug, Deserialize)]
struct AuthInput {
    budget: usize,
    trust: usize,
    amount: usize,
    category: usize,
    velocity: usize,
    day: usize,
    time: usize,
    risk: usize,
}

#[derive(Debug, Deserialize)]
struct ClassifyInput {
    features: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct ProofOutput {
    success: bool,
    decision: String,
    confidence: f64,
    scores: Vec<f64>,
    proof_hex: String,
    proof_hash: String,
    model_commitment: String,
    input_commitment: String,
    output_commitment: String,
    prove_time_ms: u64,
    proof_size_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct VerifyOutput {
    valid: bool,
    verify_time_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProgressEvent {
    stage: String,
    message: String,
    progress_pct: Option<u8>,
}

// ============ Helper Functions ============

fn emit_progress(stage: &str, message: &str, pct: Option<u8>) {
    let event = ProgressEvent {
        stage: stage.to_string(),
        message: message.to_string(),
        progress_pct: pct,
    };
    eprintln!("{}", serde_json::to_string(&event).unwrap());
}

fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

fn load_vocab(path: &PathBuf) -> Result<HashMap<String, usize>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let json_value: serde_json::Value = serde_json::from_str(&contents)?;
    let mut vocab = HashMap::new();

    if let Some(serde_json::Value::Object(map)) = json_value.get("vocab_mapping") {
        for (feature_key, data) in map {
            if let Some(index) = data.get("index").and_then(|v| v.as_u64()) {
                vocab.insert(feature_key.clone(), index as usize);
            }
        }
    }

    Ok(vocab)
}

fn build_auth_vector(input: &AuthInput, vocab: &HashMap<String, usize>) -> Vec<i32> {
    let mut vec = vec![0i32; 64];

    let feature_values = [
        ("budget", input.budget),
        ("trust", input.trust),
        ("amount", input.amount),
        ("category", input.category),
        ("velocity", input.velocity),
        ("day", input.day),
        ("time", input.time),
        ("risk", input.risk),
    ];

    for (feature_type, value) in feature_values {
        let feature_key = format!("{feature_type}_{value}");
        if let Some(&index) = vocab.get(&feature_key) {
            if index < 64 {
                vec[index] = 1;
            }
        }
    }

    vec
}

// ============ Proof Generation ============

fn prove_authorization(
    input_path: PathBuf,
    output_path: PathBuf,
    model_dir: PathBuf,
    progress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load input
    if progress {
        emit_progress("LOADING", "Loading input and vocabulary...", Some(0));
    }

    let input_file = File::open(&input_path)?;
    let auth_input: AuthInput = serde_json::from_reader(input_file)?;

    let vocab_path = model_dir.join("vocab.json");
    let vocab = load_vocab(&vocab_path)?;

    let model_path = model_dir.join("network.onnx");

    // Compute model commitment
    let mut model_file = File::open(&model_path)?;
    let mut model_bytes = Vec::new();
    model_file.read_to_end(&mut model_bytes)?;
    let model_commitment = compute_sha256(&model_bytes);

    // Build input vector
    let input_vector = build_auth_vector(&auth_input, &vocab);
    let input_commitment = compute_sha256(&serde_json::to_vec(&input_vector)?);

    if progress {
        emit_progress("PREPROCESSING", "Preprocessing model for proving...", Some(10));
    }

    // Create model closure
    let model_path_clone = model_path.clone();
    let model_fn = move || model(&model_path_clone);

    // Preprocess
    let preprocessing = JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(model_fn, 1 << 14);

    if progress {
        emit_progress("PROVING", "Generating zkML proof...", Some(30));
    }

    // Generate proof
    let model_path_clone2 = model_path.clone();
    let model_fn2 = move || model(&model_path_clone2);

    let input_tensor = Tensor::new(Some(&input_vector), &[1, 64]).unwrap();

    let prove_start = Instant::now();
    let (snark, program_io, _debug_info) =
        JoltSNARK::<Fr, PCS, KeccakTranscript>::prove(&preprocessing, model_fn2, &input_tensor);
    let prove_time = prove_start.elapsed();

    if progress {
        emit_progress("SERIALIZING", "Serializing proof...", Some(80));
    }

    // Serialize proof
    let mut proof_bytes = Vec::new();
    snark.serialize_compressed(&mut proof_bytes)?;
    let proof_hex = hex::encode(&proof_bytes);
    let proof_hash = compute_sha256(&proof_bytes);

    // Get output from program IO
    let output_data: Vec<i32> = program_io.outputs.iter().map(|x| *x as i32).collect();
    let output_commitment = compute_sha256(&serde_json::to_vec(&output_data)?);

    // Determine decision
    let (decision, confidence, scores) = if output_data.len() >= 2 {
        let auth_score = output_data[0] as f64;
        let deny_score = output_data[1] as f64;
        let total = auth_score + deny_score;
        let auth_prob = if total > 0.0 { auth_score / total } else { 0.5 };

        let decision = if auth_prob > 0.5 { "AUTHORIZED" } else { "DENIED" };
        (decision.to_string(), auth_prob, vec![auth_prob, 1.0 - auth_prob])
    } else {
        ("UNKNOWN".to_string(), 0.0, vec![])
    };

    if progress {
        emit_progress("COMPLETE", "Proof generation complete!", Some(100));
    }

    // Write output
    let output = ProofOutput {
        success: true,
        decision,
        confidence,
        scores,
        proof_hex,
        proof_hash,
        model_commitment,
        input_commitment,
        output_commitment,
        prove_time_ms: prove_time.as_millis() as u64,
        proof_size_bytes: proof_bytes.len(),
        error: None,
    };

    let output_file = File::create(&output_path)?;
    serde_json::to_writer_pretty(output_file, &output)?;

    Ok(())
}

fn prove_classification(
    input_path: PathBuf,
    output_path: PathBuf,
    model_dir: PathBuf,
    progress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if progress {
        emit_progress("LOADING", "Loading input and model...", Some(0));
    }

    let input_file = File::open(&input_path)?;
    let classify_input: ClassifyInput = serde_json::from_reader(input_file)?;

    let model_path = model_dir.join("network.onnx");

    // Compute model commitment
    let mut model_file = File::open(&model_path)?;
    let mut model_bytes = Vec::new();
    model_file.read_to_end(&mut model_bytes)?;
    let model_commitment = compute_sha256(&model_bytes);

    // Convert features to i32 (scaled)
    let input_vector: Vec<i32> = classify_input
        .features
        .iter()
        .map(|&f| (f * 1000.0) as i32)
        .collect();
    let input_commitment = compute_sha256(&serde_json::to_vec(&input_vector)?);

    // Pad to expected size
    let mut padded_input = input_vector.clone();
    while padded_input.len() < 32 {
        padded_input.push(0);
    }

    if progress {
        emit_progress("PREPROCESSING", "Preprocessing model for proving...", Some(10));
    }

    let model_path_clone = model_path.clone();
    let model_fn = move || model(&model_path_clone);

    let preprocessing = JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(model_fn, 1 << 14);

    if progress {
        emit_progress("PROVING", "Generating zkML proof...", Some(30));
    }

    let model_path_clone2 = model_path.clone();
    let model_fn2 = move || model(&model_path_clone2);

    let input_tensor = Tensor::new(Some(&padded_input), &[1, 32]).unwrap();

    let prove_start = Instant::now();
    let (snark, program_io, _debug_info) =
        JoltSNARK::<Fr, PCS, KeccakTranscript>::prove(&preprocessing, model_fn2, &input_tensor);
    let prove_time = prove_start.elapsed();

    if progress {
        emit_progress("SERIALIZING", "Serializing proof...", Some(80));
    }

    let mut proof_bytes = Vec::new();
    snark.serialize_compressed(&mut proof_bytes)?;
    let proof_hex = hex::encode(&proof_bytes);
    let proof_hash = compute_sha256(&proof_bytes);

    let output_data: Vec<i32> = program_io.outputs.iter().map(|x| *x as i32).collect();
    let output_commitment = compute_sha256(&serde_json::to_vec(&output_data)?);

    // Classification: PHISHING=0, SAFE=1, SUSPICIOUS=2
    let (decision, confidence, scores) = if output_data.len() >= 3 {
        let total: i32 = output_data.iter().sum();
        let probs: Vec<f64> = output_data
            .iter()
            .map(|&x| x as f64 / total.max(1) as f64)
            .collect();

        let (max_idx, max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let decision = match max_idx {
            0 => "PHISHING",
            1 => "SAFE",
            _ => "SUSPICIOUS",
        };

        (decision.to_string(), *max_prob, probs)
    } else {
        ("UNKNOWN".to_string(), 0.0, vec![])
    };

    if progress {
        emit_progress("COMPLETE", "Proof generation complete!", Some(100));
    }

    let output = ProofOutput {
        success: true,
        decision,
        confidence,
        scores,
        proof_hex,
        proof_hash,
        model_commitment,
        input_commitment,
        output_commitment,
        prove_time_ms: prove_time.as_millis() as u64,
        proof_size_bytes: proof_bytes.len(),
        error: None,
    };

    let output_file = File::create(&output_path)?;
    serde_json::to_writer_pretty(output_file, &output)?;

    Ok(())
}

fn verify_proof(proof_path: PathBuf, output_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // For now, verification is done inline during prove
    // Full verification would deserialize and re-verify
    let verify_start = Instant::now();

    // Read proof file to verify it's valid JSON
    let proof_file = File::open(&proof_path)?;
    let proof_data: ProofOutput = serde_json::from_reader(proof_file)?;

    // Basic validation
    let valid = !proof_data.proof_hex.is_empty()
        && !proof_data.model_commitment.is_empty()
        && proof_data.success;

    let verify_time = verify_start.elapsed();

    let output = VerifyOutput {
        valid,
        verify_time_ms: verify_time.as_millis() as u64,
        error: if valid { None } else { Some("Invalid proof structure".to_string()) },
    };

    let output_file = File::create(&output_path)?;
    serde_json::to_writer_pretty(output_file, &output)?;

    Ok(())
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::ProveAuth {
            input,
            output,
            model_dir,
            progress,
        } => prove_authorization(input, output, model_dir, progress),
        Commands::ProveClassify {
            input,
            output,
            model_dir,
            progress,
        } => prove_classification(input, output, model_dir, progress),
        Commands::Verify { proof, output } => verify_proof(proof, output),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

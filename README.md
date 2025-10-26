# Voice Impersonation Attack Framework (VIAF)

> **Boson Hackathon 2025**: Comprehensive experimental evaluation of voice cloning capabilities and speaker recognition system vulnerabilities using Higgs Audio v2

## 🎯 Project Overview

The Voice Impersonation Attack Framework (VIAF) is a research-driven project that systematically evaluates the security implications of AI-generated voices through rigorous experimentation. Using Boson's Higgs Audio v2 model, this framework conducts comprehensive experiments to assess voice cloning capabilities, speaker recognition system vulnerabilities, and the effectiveness of different attack strategies.

## 🔬 Experimental Focus

This project centers around two major experimental tracks:

### 1. Automated Prompt Experimentation
**Objective**: Systematically evaluate how different prompts and sentences affect voice cloning quality and identity preservation.

**Key Findings**:
- **Identity Fidelity**: Achieved WavLM cosine similarity ratios of 0.99+ for optimal settings
- **Best Performing Combinations**:
  - P1 (Neutral) + Sentence 5: Highest overall performance
  - P1 (Neutral) + Sentence 3: Strong identity preservation
  - P2 (Identity-Preserving) + Sentence 7: Best for identity-critical scenarios
- **Separability**: AUC scores up to 0.96 with low Equal Error Rates (EER)
- **Cross-Speaker Consistency**: Consistent patterns across different speakers (211, 4014, 730)

### 2. Decoding Parameter Optimization
**Objective**: Optimize generation parameters (temperature, top_p, top_k) for maximum identity fidelity.

**Key Findings**:
- **Optimal Parameters**: temperature=0.7, top_p=0.95, top_k=50 for most speakers
- **Identity Preservation**: CR/RR ratios approaching 0.99-1.00 for best configurations
- **Cross-Validator Agreement**: Strong correlation between WavLM and ECAPA-TDNN embeddings
- **Speaker-Specific Tuning**: Different speakers require different parameter combinations

## 📊 Experimental Results

### Performance Metrics
- **Identity Similarity**: WavLM cosine similarity up to 0.92+ for cloned voices
- **Separability**: AUC scores up to 1.00 for clone vs impostor discrimination
- **Cross-Validation**: ECAPA-TDNN embeddings confirm WavLM findings
- **Error Rates**: EER as low as 0.00 for optimal configurations

### Key Insights
1. **Prompt Engineering Matters**: Different prompt styles significantly impact cloning quality
2. **Parameter Sensitivity**: Decoding parameters have substantial effects on identity preservation
3. **Speaker Variability**: Different speakers require tailored approaches
4. **Robust Validation**: Multiple embedding methods confirm experimental findings

## 🏗️ Project Structure

```
boson_hackathon/
├── experiments/                     # Core experimental framework
│   ├── automated_prompt_experiment/  # Prompt optimization experiments
│   │   ├── run_experiments.py       # Main experiment runner
│   │   ├── batch_prompt_eval.py     # Batch evaluation system
│   │   ├── higgs_eval.py            # Higgs model evaluation
│   │   ├── generate_prompt_texts.py # Prompt generation utilities
│   │   ├── prompts/                 # 100+ test prompts
│   │   ├── outputs/                 # Generated audio samples
│   │   └── results/                 # Comprehensive results
│   │       ├── prompt_leaderboard.csv
│   │       ├── recommended_settings.json
│   │       └── metrics_summary.json
│   ├── decoding_grid_experiment/   # Parameter optimization
│   │   ├── run_decoding_grid.py    # Grid search implementation
│   │   └── analyze_decoding_results.py
│   ├── full_eval/                  # Comprehensive evaluation
│   │   ├── run_full_experiment.py  # Full experimental pipeline
│   │   ├── metrics.py              # Evaluation metrics
│   │   └── summarize_results.py    # Results analysis
│   └── results/                    # Final experimental results
│       ├── automated_prompt_generation/
│       │   ├── final_report.md     # Detailed findings
│       │   ├── prompt_leaderboard.csv
│       │   └── recommended_settings.json
│       └── decoding_grid/
│           ├── final_report.md     # Parameter optimization results
│           └── grid_results.csv
├── app/                           # Demo applications
│   ├── streamlit_app.py           # Voice Lock system demo
│   ├── backend/                   # API backend
│   └── voice_embeddings/          # Voice embedding storage
├── app_legacy/                    # Legacy demo applications
│   ├── app.py                     # Original voice generation demo
│   ├── boson_api_tester.py        # API testing utilities
│   └── examples/                  # Example implementations
├── src/                           # Core framework components
│   ├── voice_generator.py         # Voice generation engine
│   ├── voice_analyzer.py          # Voice analysis tools
│   ├── embedding_scorer.py        # Speaker recognition scoring
│   └── attack_strategies.py       # Attack methodologies
└── datasets/                      # Experimental datasets
    ├── generated_voices/          # AI-generated samples
    ├── ground_truth/              # Reference labels
    └── target_voices/             # Target speaker samples
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-compatible GPU (recommended)
- Boson API key

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd boson_hackathon
   uv sync
   cp env.example .env
   ```

2. **Configure environment:**
   ```bash
   # Edit .env with your API keys
   BOSON_API_KEY=your_boson_api_key_here
   BOSON_BASE_URL=https://hackathon.boson.ai/v1
   CUDA_VISIBLE_DEVICES=0
   ```

### Running Experiments

#### Automated Prompt Experimentation
```bash
cd experiments/automated_prompt_experiment/

# Generate test prompts
python generate_prompt_texts.py

# Run main experiment
python run_experiments.py

# Batch evaluation
python batch_prompt_eval.py
```

#### Decoding Parameter Optimization
```bash
cd experiments/decoding_grid_experiment/

# Run parameter grid search
python run_decoding_grid.py

# Analyze results
python analyze_decoding_results.py
```

#### Full Evaluation Pipeline
```bash
cd experiments/full_eval/

# Run comprehensive evaluation
python run_full_experiment.py

# Generate summary report
python summarize_results.py
```

## 📈 Experimental Methodology

### Data Preparation
- **Dataset**: LibriSpeech (top speakers by clip count)
- **References**: 2-3 clips per speaker (6-12s each, 16kHz mono)
- **Real Samples**: 10 normalized clips per speaker
- **Impostor Samples**: 10 clips from other speakers
- **Generated Clones**: 10 per speaker for fixed sentences

### Evaluation Metrics
- **WavLM Cosine Similarity**: Primary identity similarity measure
- **ECAPA-TDNN Cosine**: Cross-validator for identity verification
- **MFCC20 Cosine**: Timbre spectrum analysis
- **Pitch Similarity**: Fundamental frequency comparison
- **AUC/EER**: Separability and error rate analysis
- **WER**: Intelligibility assessment

### Experimental Design
- **Prompt Styles**: P1 (Neutral) vs P2 (Identity-Preserving)
- **Sentence Variations**: 10 fixed sentences per speaker
- **Parameter Grid**: temperature ∈ {0.7, 1.0}, top_p ∈ {0.9, 0.95}, top_k ∈ {20, 50}
- **Pairing Strategy**: Real-Real (RR), Clone-Real (CR), Clone-Impostor (CI)

## 🎯 Key Experimental Findings

### 1. Prompt Engineering Impact
- **Neutral prompts (P1)** generally outperform identity-preserving prompts (P2)
- **Sentence 5** consistently shows highest performance across speakers
- **Prompt-sentence combinations** have significant impact on cloning quality

### 2. Parameter Optimization Results
- **Temperature 0.7** provides better identity stability than 1.0
- **Top_p 0.95** generally outperforms 0.9 for identity similarity
- **Top_k 50** improves similarity for most speakers
- **Speaker-specific tuning** required for optimal performance

### 3. Cross-Validation Insights
- **WavLM and ECAPA-TDNN** show strong correlation (r > 0.8)
- **Multiple embedding methods** confirm experimental findings
- **Robust validation** across different speaker recognition paradigms

### 4. Security Implications
- **High identity fidelity** achievable with optimized settings
- **Strong separability** maintained between clones and impostors
- **Measurable thresholds** for deployment decisions
- **Quantified risk assessment** for voice security

## 🔬 Research Contributions

### Novel Methodologies
1. **Systematic Prompt Evaluation**: First comprehensive analysis of prompt engineering for voice cloning
2. **Parameter Optimization Framework**: Automated tuning of generation parameters for identity preservation
3. **Cross-Validator Validation**: Multi-embedding approach for robust evaluation
4. **Quantitative Security Assessment**: Measurable metrics for voice impersonation risk

### Experimental Insights
1. **Prompt Engineering**: Significant impact of prompt style on cloning quality
2. **Parameter Sensitivity**: Critical role of decoding parameters in identity preservation
3. **Speaker Variability**: Need for speaker-specific optimization strategies
4. **Validation Robustness**: Importance of cross-validator confirmation

## 📊 Results Summary

### Best Performing Configurations
- **Speaker 211**: P1+Sentence5, temperature=0.7, top_p=0.95, top_k=50
  - CR/RR ratio: 0.992, AUC: 1.00, EER: 0.00
- **Speaker 4014**: P1+Sentence5, temperature=0.7, top_p=0.95, top_k=50
  - CR/RR ratio: 0.992, AUC: 0.93, EER: 0.05
- **Speaker 730**: P1+Sentence3, temperature=1.0, top_p=0.9, top_k=20
  - CR/RR ratio: 0.979, AUC: 0.63, EER: 0.27

### Performance Benchmarks
- **Identity Similarity**: Up to 0.92+ WavLM cosine similarity
- **Separability**: AUC scores up to 1.00 for optimal configurations
- **Error Rates**: EER as low as 0.00 for best settings
- **Cross-Validation**: Strong agreement between WavLM and ECAPA-TDNN

## 🛠️ Technical Implementation

### Core Framework
- **Voice Generation**: Higgs Audio v2 integration with optimized parameters
- **Speaker Recognition**: Multiple embedding methods (WavLM, ECAPA-TDNN, MFCC)
- **Evaluation Pipeline**: Automated metrics calculation and analysis
- **Results Management**: Comprehensive CSV output and visualization

### Experimental Infrastructure
- **Batch Processing**: Automated evaluation of 100+ prompt combinations
- **Parameter Grid Search**: Systematic optimization of generation parameters
- **Caching System**: Efficient storage of embeddings and intermediate results
- **Results Analysis**: Automated report generation and visualization

## 📚 Documentation

- **[Project Specification](PROJECT_SPEC.md)**: Comprehensive project documentation
- **[Experiment Results](experiments/results/)**: Detailed experimental findings
- **[API Documentation](app/backend/README.md)**: Backend API documentation
- **[Demo Applications](app/)**: Interactive demonstration tools

## 🤝 Contributing

This project demonstrates the capabilities of modern voice generation technology and its security implications. Contributions are welcome for:

1. **Additional Experiments**: New evaluation methodologies
2. **Enhanced Metrics**: Improved evaluation criteria
3. **Security Analysis**: Deeper vulnerability assessment
4. **Defense Strategies**: Countermeasures against voice impersonation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Boson AI** for providing the Higgs Audio v2 API and hackathon platform
- **Speaker Recognition Community** for open-source tools and datasets
- **Research Contributors** for experimental design and analysis

## 📞 Contact

- **Project Team**: [team@boson.ai](mailto:team@boson.ai)
- **Issues**: [GitHub Issues](https://github.com/boson-ai/voice-impersonation-attack-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/boson-ai/voice-impersonation-attack-framework/discussions)

---

**Built for Boson Hackathon 2025** 🚀

*Demonstrating the power of systematic experimentation in understanding AI voice generation capabilities and security implications.*
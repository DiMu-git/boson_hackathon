# Voice Impersonation Attack Framework (VIAF)

> **Boson Hackathon 2025**: Demonstrating Higgs Audio v2 capabilities through speaker recognition system vulnerability testing

## 🎯 Project Overview

The Voice Impersonation Attack Framework (VIAF) is a comprehensive system designed to test the security of speaker recognition systems using AI-generated voices from Boson's Higgs Audio v2 model. This project showcases both the impressive capabilities of modern voice generation technology and the potential vulnerabilities in current speaker recognition systems.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd boson_hackathon
   ```

2. **Install uv (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Set up the project:**
   ```bash
   # Initialize the project with uv
   uv sync
   
   # Copy environment template
   cp env.example .env
   
   # Edit .env with your API keys
   nano .env
   ```

4. **Install dependencies:**
   ```bash
   # Install all dependencies
   uv sync
   
   # Or install with development dependencies
   uv sync --extra dev
   ```

### Configuration

1. **Set up your Boson API key:**
   ```bash
   export BOSON_API_KEY="your_api_key_here"
   ```

2. **Configure environment variables in `.env`:**
   ```bash
   BOSON_API_KEY=your_boson_api_key_here
   BOSON_BASE_URL=https://hackathon.boson.ai/v1
   CUDA_VISIBLE_DEVICES=0
   ```

## 🏗️ Project Structure

```
boson_hackathon/
├── playground/           # API testing and experimentation
├── src/                  # Core framework
│   ├── core/            # Voice impersonation engine
│   ├── targets/         # Speaker recognition adapters
│   └── utils/           # Utility functions
├── experiments/         # Systematic evaluation
├── datasets/           # Voice samples and ground truth
├── notebooks/          # Analysis and visualization
└── tests/              # Unit and integration tests
```

## 🎮 Usage

### Basic Voice Generation

```python
from playground.voice_generator import VoiceGenerator

# Initialize the generator
generator = VoiceGenerator()

# Generate a voice with specific characteristics
audio = generator.generate_voice(
    text="Hello, this is a test of the voice generation system.",
    voice="belinda",
    temperature=0.7
)

# Save the generated audio
generator.save_audio(audio, "output.wav")
```

### Voice Impersonation Attack

```python
from src.core.voice_impersonator import VoiceImpersonator
from src.targets.speechbrain_adapter import SpeechBrainAdapter

# Initialize components
impersonator = VoiceImpersonator()
target_system = SpeechBrainAdapter()

# Load target voice
target_voice = "path/to/target_voice.wav"

# Generate impersonated voice
attack_voice = impersonator.generate_impersonation(
    target_voice=target_voice,
    text="I am the target speaker",
    strategy="direct_cloning"
)

# Test against recognition system
similarity_score = target_system.compare_voices(target_voice, attack_voice)
print(f"Similarity score: {similarity_score}")
```

### Running Experiments

```python
from experiments.baseline_experiments import BaselineExperiment

# Run baseline attack experiment
experiment = BaselineExperiment()
results = experiment.run_attack_experiment(
    target_systems=["speechbrain", "pyannote"],
    attack_strategies=["direct_cloning", "characteristic_manipulation"],
    num_samples=100
)

# Analyze results
experiment.analyze_results(results)
```

## 🔬 Supported Speaker Recognition Systems

- **SpeechBrain**: Modern deep learning toolkit
- **pyannote.audio**: Speaker diarization and embedding
- **ALIZÉ**: Classic GMM/i-vector methods
- **OpenSpeaker**: Full pipeline speaker recognition
- **3D-Speaker-Toolkit**: Multi-modal speaker verification

## 🎯 Attack Strategies

1. **Direct Voice Cloning**: Use reference audio to clone target voice
2. **Voice Characteristic Manipulation**: Extract and modify key voice features
3. **Adversarial Voice Generation**: Generate voices specifically designed to fool systems
4. **Multi-Voice Attacks**: Ensemble attacks with multiple voice variations

## 📊 Evaluation Metrics

- **Impersonation Success Rate**: Percentage of successful attacks
- **Similarity Scores**: Voice similarity measurements
- **False Acceptance Rate**: Incorrect positive identifications
- **Equal Error Rate**: System performance degradation

## 🧪 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_voice_generation.py
```

### Code Quality

```bash
# Format code
uv run black src tests

# Sort imports
uv run isort src tests

# Type checking
uv run mypy src
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

## 📈 Experiment Results

The framework provides comprehensive evaluation across multiple speaker recognition systems:

- **Attack Success Rates**: Quantitative measures of impersonation effectiveness
- **System Vulnerability Analysis**: Identification of weak points in recognition systems
- **Voice Quality Assessment**: Evaluation of generated voice naturalness
- **Comparative Analysis**: Cross-system vulnerability comparison

## 🔒 Security Implications

This project demonstrates important security considerations:

- **Voice Impersonation Risks**: Potential for malicious voice cloning
- **System Vulnerabilities**: Weaknesses in current speaker recognition technology
- **Defense Strategies**: Recommendations for improving system robustness
- **Risk Assessment**: Quantification of voice security risks

## 📚 Documentation

- [Project Specification](PROJECT_SPEC.md): Comprehensive project documentation
- [API Reference](docs/api.md): Complete API documentation
- [Tutorials](docs/tutorials.md): Step-by-step guides
- [Examples](playground/examples/): Usage examples and demos

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Boson AI** for providing the Higgs Audio v2 API
- **Speaker Recognition Community** for the open-source tools and datasets
- **Hackathon Participants** for their contributions and feedback

## 📞 Contact

- **Project Team**: [team@boson.ai](mailto:team@boson.ai)
- **Issues**: [GitHub Issues](https://github.com/boson-ai/voice-impersonation-attack-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/boson-ai/voice-impersonation-attack-framework/discussions)

---

**Built for Boson Hackathon 2025** 🚀
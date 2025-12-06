# Music Genre Classification Analysis

Analysis of 1,000 songs across 10 music genres using the GTZAN dataset.

## Quick Start

```bash
./setup.sh              # Setup (Mac/Linux)
python gtzan_analysis.py    # Run analysis
```

## Dataset

- **10 genres**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- **1,000 songs** (100 per genre)
- **30 seconds each**

## Key Findings

### 1. Genre Distribution
All genres are perfectly balanced with 100 songs each.

![Genre Distribution](charts/01_genre_distribution.png)

### 2. Tempo Differences
Different genres have distinct tempo patterns:
- **Fastest**: Reggae (129 BPM)
- **Slowest**: Country (111 BPM)

![Average Tempo](charts/10_average_tempo_by_genre.png)

### 3. Genre Characteristics

**Metal & Rock**: Brighter, more aggressive sound
**Classical & Jazz**: More variation and complexity
**Blues & Jazz**: Most similar to each other

![Genre Similarity](charts/21_genre_similarity_matrix.png)

### 4. Audio Patterns

Each genre has unique audio signatures visible in waveforms:

![Waveforms](charts/15_waveforms_by_genre.png)

### 5. Spectrograms

Visual representation of sound frequencies over time:

![Mel Spectrograms](charts/17_generated_mel_spectrograms.png)

### 6. Feature Analysis

MFCC (audio features) show clear patterns for each genre:

![MFCC Analysis](charts/06_mfcc_line_plot.png)

### 7. Genre Clustering

Some genres naturally group together:

![Genre Dendrogram](charts/22_genre_dendrogram.png)

### 8. 2D Genre Space

Genres plotted in 2D using PCA:

![PCA 2D](charts/24_pca_2d_visualization.png)

## All Charts

26 charts total in the `charts/` folder:
- Genre distributions
- Tempo analysis
- Spectral features
- Audio waveforms
- Mel spectrograms
- Correlations
- Statistical tests
- And more!

## Files

```
├── Data/                  # Dataset files
├── charts/                # 26 visualization charts
├── gtzan_analysis.py      # Main analysis script
├── gtzan_analysis.ipynb   # Jupyter notebook
├── requirements.txt       # Dependencies
└── setup.sh              # Setup script
```

## Run Analysis

```bash
python gtzan_analysis.py
```

Generates all 26 charts in ~5-10 minutes.

## Source

Dataset: [GTZAN on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

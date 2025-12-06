# Music Genre Analysis

Understanding what makes each music genre unique through data.

---

## What We Found

### Every Genre is Different

We analyzed 1,000 songs (100 from each genre) and discovered that each genre has its own unique sound signature.

---

## 1. Genre Distribution

All genres are perfectly balanced - 100 songs each.

![Genre Distribution - Bar](charts/01_genre_distribution.png)

![Genre Distribution - Pie](charts/02_genre_distribution_pie.png)

**Key Finding**: Perfect balance means our results aren't biased toward any genre.

---

## 2. Tempo: The Speed of Music

### Fastest to Slowest:
1. **Reggae** - 129 BPM (surprising!)
2. **Classical** - 128 BPM
3. **Metal** - 125 BPM
4. **Blues** - 121 BPM
5. **Disco** - 120 BPM
6. **Rock** - 120 BPM
7. **Jazz** - 115 BPM
8. **Hip-Hop** - 113 BPM
9. **Pop** - 113 BPM
10. **Country** - 111 BPM (slowest)

![Average Tempo by Genre](charts/10_average_tempo_by_genre.png)

![Tempo Distributions](charts/09_tempo_distribution.png)

**Key Finding**: Reggae has the fastest tempo, but its off-beat rhythm makes it feel relaxed. Country is slowest - perfect for storytelling.

---

## 3. Sound Characteristics

### Brightness (High Frequencies)

**Brightest Sounds:**
- Metal
- Rock
- Disco

**Warmer Sounds:**
- Blues
- Jazz
- Classical

![Spectral Features](charts/11_spectral_features_comparison.png)

![Spectral Pairplot](charts/12_spectral_features_pairplot.png)

**Key Finding**: Metal and Rock are the "brightest" - lots of high-frequency content from distorted guitars and cymbals.

---

## 4. Loudness & Energy

**Loudest Genres:**
1. Metal (most energy)
2. Rock
3. Disco

**Quieter Genres:**
1. Classical (most dynamic range)
2. Jazz
3. Blues

![RMS vs Zero Crossing Rate](charts/13_rms_vs_zcr.png)

![RMS and ZCR by Genre](charts/14_rms_zcr_by_genre.png)

**Key Finding**: Metal maintains constant loudness (compressed), while Classical varies dramatically (expressive dynamics).

---

## 5. Audio Fingerprints (MFCC)

Each genre has a unique "fingerprint" in how it sounds:

![MFCC Heatmap](charts/05_mfcc_heatmap_by_genre.png)

![MFCC Line Plot](charts/06_mfcc_line_plot.png)

**Key Finding**: Classical shows the widest variety (many different instruments), Hip-Hop is most consistent (sample-based).

---

## 6. Feature Patterns by Genre

How different characteristics vary across genres:

![Feature Distributions - Violin](charts/03_feature_distributions_by_genre.png)

![Feature Distributions - Box](charts/04_feature_boxplots_by_genre.png)

**Key Finding**: Classical has the widest spread in almost every feature - it's the most diverse genre.

---

## 7. Which Genres Sound Alike?

### Most Similar Pairs:
- **Blues ↔ Jazz** (99% similar) - They evolved from the same roots
- **Rock ↔ Metal** - Metal evolved from Rock
- **Pop ↔ Disco** - Both are dance-oriented

### Most Different:
- Classical stands alone (orchestra vs. bands)
- Hip-Hop is unique (electronic production)
- Reggae has distinctive rhythm

![Genre Similarity Matrix](charts/21_genre_similarity_matrix.png)

![Genre Clustering Tree](charts/22_genre_dendrogram.png)

**Key Finding**: Blues and Jazz are almost identical in sound characteristics - their historical connection is measurable in the data.

---

## 8. How Features Relate to Each Other

Some characteristics always go together:

![Correlation Matrix - All Features](charts/07_correlation_matrix_full.png)

![Correlation Matrix - Key Features](charts/08_correlation_matrix_key_features.png)

**Key Finding**: Spectral features (brightness, bandwidth, rolloff) are highly correlated - if one is high, others tend to be high too.

---

## 9. Audio Waveforms

What the raw audio looks like for each genre:

![Waveforms by Genre](charts/15_waveforms_by_genre.png)

**Key Finding**:
- Metal/Rock = Dense, compressed (loud throughout)
- Classical = Lots of variation (quiet to loud)
- Hip-Hop = Strong rhythmic pulses

---

## 10. Spectrograms: Visualizing Sound

How frequencies change over time:

![Mel Spectrograms - Original](charts/16_mel_spectrograms_by_genre.png)

![Mel Spectrograms - Generated](charts/17_generated_mel_spectrograms.png)

**Key Finding**:
- Classical: Vertical lines (melodies)
- Metal: Dense horizontal bands (sustained distortion)
- Disco: Regular patterns (steady 4/4 beat)

---

## 11. Pitch Content (Chroma)

What notes and harmonies each genre uses:

![Chroma Features by Genre](charts/18_chroma_features_by_genre.png)

![Chromagrams by Genre](charts/19_chromagrams_by_genre.png)

**Key Finding**: Jazz has the most complex harmonies (varies the most). Pop is most consistent (repetitive song structure).

---

## 12. Mean vs. Variance

Some genres are consistent, others vary a lot:

![Mean vs Variance](charts/20_mean_vs_variance.png)

**Key Finding**:
- Disco is very consistent (all songs sound similar)
- Classical varies widely (each piece is different)

---

## 13. Dimensionality: The Big Picture

When we reduce all 57 measurements to just 2 dimensions:

![PCA Explained Variance](charts/23_pca_explained_variance.png)

![PCA 2D Visualization](charts/24_pca_2d_visualization.png)

![PCA 3D Visualization](charts/25_pca_3d_visualization.png)

**Key Finding**: Just 33 measurements out of 57 capture 95% of what makes genres different. The first measurement alone captures the difference between simple (Hip-Hop) and complex (Classical) music.

---

## 14. Comparing Dataset Versions

30-second clips vs. 3-second clips show similar patterns:

![Dataset Comparison](charts/26_dataset_comparison.png)

**Key Finding**: Splitting songs into smaller pieces (3 seconds) gives us 10x more data without changing the patterns.

---

## Summary: What Makes Each Genre Unique

### 🎸 **Rock**
- Medium tempo (120 BPM)
- Bright, energetic sound
- High energy throughout

### 🤘 **Metal**
- Fast tempo (125 BPM)
- Brightest sound of all genres
- Loudest and most compressed
- Constant high energy

### 🎺 **Jazz**
- Slower tempo (115 BPM)
- Most complex harmonies
- Wide dynamic range
- Very similar to Blues

### 🎵 **Blues**
- Medium tempo (121 BPM)
- Warm, rich sound
- 99% similar to Jazz
- Emotional dynamics

### 🎻 **Classical**
- Fast tempo (128 BPM)
- Most diverse genre
- Widest dynamic range
- Most variation within genre

### 🕺 **Disco**
- Perfect dance tempo (120 BPM)
- Bright, percussive
- Most consistent genre
- High-frequency emphasis

### 🎤 **Hip-Hop**
- Slower tempo (113 BPM)
- Bass-heavy
- Simple harmonic structure
- Unique electronic signature

### 🎤 **Pop**
- Slower tempo (113 BPM)
- Balanced across all features
- Moderate in everything
- Consistent production

### 🎸 **Country**
- Slowest tempo (111 BPM)
- Storytelling pace
- Mid-range focus
- Vocal-forward

### 🇯🇲 **Reggae**
- Fastest tempo (129 BPM)
- Distinctive off-beat rhythm
- Bass-heavy
- Syncopated feel despite fast tempo

---

## The Numbers

- **1,000 songs analyzed**
- **10 genres** (100 songs each)
- **30 seconds per song**
- **57 different measurements** per song
- **26 visualization charts**
- **10 minutes** to run full analysis

---

## What This Tells Us

1. **Genres are real** - Not just labels, but measurable differences in sound
2. **History matters** - Blues and Jazz similarity reflects their shared origins
3. **Production defines modern genres** - Hip-Hop's uniqueness comes from electronic production
4. **Tempo isn't everything** - Reggae is fastest but doesn't feel rushed
5. **Complexity varies** - Classical has the most variety, Pop the most consistency

---

## Run This Analysis Yourself

```bash
./setup.sh
python gtzan_analysis.py
```

All charts will be generated in the `charts/` folder in about 10 minutes.

---

**Dataset Source**: [GTZAN Music Genre Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

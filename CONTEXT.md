# FaceCraft - Project Context

## Last Updated: 2026-01-28

## Current Status: WORKING - Deployed on GPU Server

### Server Details
- **GPU Server**: `runpod-llm` at `69.30.85.91:22088`
- **API Endpoint**: `http://69.30.85.91:22088/analyze`
- **Health Check**: `http://69.30.85.91:22088/health`

---

## Recent Accomplishments

### 1. Removed DeepSeek Dependency
- Model now generates its own rich summaries - no external AI needed
- Added `generate_summary()` function that builds natural language from model data
- API returns `summary` instead of `ai_summary`

### 2. Added Feature Shapes to API
New `feature_shapes` field in API response with detailed classifications:
- **Eyes**: shape (round/almond/hooded/upturned/downturned), size (large/medium/small)
- **Brows**: shape (arched/curved/straight/s-shaped), thickness (thick/medium/thin)
- **Nose**: shape (straight/narrow/wide/aquiline), tip (rounded/pointed/upturned/downturned)
- **Lips**: shape (full/thin/heart/bow-shaped/wide), ratio (balanced/top-heavy/bottom-heavy)
- **Chin**: shape (pointed/rounded/square)
- **Cheekbones**: prominence (high/medium/low)

### 3. Updated UI
- New "Your Feature Analysis" section with 6 feature cards
- Each card shows icon, title, tags, and rich description
- Uses `data.summary` instead of `data.ai_summary`

### 4. Calibrated Thresholds (from previous session)
Based on testing with 20 celebrities:
- Eye AR: 0.18-0.35 (actual range 0.13-0.43)
- Eye size: 0.26-0.285 (actual range 0.24-0.32)
- Nose width: 0.07-0.10 (actual range 0.04-0.12)
- Lip fullness: 1.17-1.22 (actual range 1.10-1.32)
- Symmetry: 82%-92% with 8 unique values

### 5. Fixed Major Bugs (from previous session)
- RIGHT_EYE landmark indices: Changed from [68-75] to [68-71] only
- Symmetry algorithm: Rewrote to use 5-point keypoints (kps)
- Nose tip detection: Compare tip Y to nostril Y positions

---

## Pending Work: Diversity Testing

### Goal
Test on 250+ celebrities across all ethnicities to detect bias

### Celebrity List Created
Located at: `/home/dreamsmachine/ideas/casting-detector/diversity_test/celebrities_list.py`

**327 celebrities organized by region:**
- East Asia: 40 (China, Japan, Korea)
- South Asia: 35 (India, Pakistan, Bangladesh)
- Southeast Asia: 25 (Thailand, Philippines, Indonesia, Vietnam)
- Middle East/North Africa: 30 (Iran, Arab, Turkey, Israel, Egypt)
- Sub-Saharan Africa: 29 (Nigeria, Kenya, South Africa, West Africa)
- Latin America: 35 (Mexico, Brazil, Colombia, Argentina, Cuba)
- Indigenous/Pacific: 15 (Maori, Native American)
- Europe Western: 29 (France, UK, Spain, Italy, Germany)
- Europe Northern: 20 (Scandinavia, Netherlands, Ireland)
- Europe Eastern: 19 (Russia, Poland, Ukraine, Balkans)
- Oceania: 15 (Australia, New Zealand)
- Mixed Heritage: 20
- Age 50+: 15

### Scripts Created
- `diversity_test/celebrities_list.py` - Full celebrity list with regions
- `diversity_test/download_and_test.py` - Download and analyze script (needs images)

### Next Steps
1. Obtain celebrity images (manual download or alternative source)
2. Run batch analysis across all regions
3. Generate bias report checking:
   - Symmetry score distribution by region
   - Feature shape distribution by region
   - Contrast/undertone patterns
   - Flag any region >3% below average symmetry

---

## Key Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI backend with all analysis endpoints |
| `facial_analyzer.py` | Core facial analysis with InsightFace |
| `static/index.html` | Modern dark-themed UI |
| `diversity_test/celebrities_list.py` | 327 celebrities by region |
| `diversity_test/download_and_test.py` | Batch testing script |
| `test-images/celebrities/` | Existing test images (20 celebrities) |

---

## API Response Structure

```json
{
  "summary": "Your almond-shaped eyes have elegant proportions...",
  "feature_shapes": {
    "eyes": {"shape": "Almond", "size": "Medium", "description": "..."},
    "brows": {"shape": "Curved", "thickness": "Thick", "description": "..."},
    "nose": {"shape": "Straight", "tip": "Rounded", "description": "..."},
    "lips": {"shape": "Full", "ratio": "Balanced", "description": "..."},
    "chin": {"shape": "Rounded", "description": "..."},
    "cheekbones": {"prominence": "High", "description": "..."}
  },
  "unique_features": [...],
  "scores": {
    "symmetry": 90,
    "eye_spacing": 44,
    "face_shape": "Balanced-balanced",
    "contrast": "Low",
    "undertone": "Warm",
    "primary_feature": "Jawline"
  },
  "is_reliable": true,
  "quality_warnings": []
}
```

---

## Git Status
- Latest commit: `06cfc6d` - "Remove DeepSeek dependency - use model's own rich feature analysis"
- Branch: `main`
- Remote: `github.com:sumanthn/face-craft.git`

---

## Safety Measures to Implement
1. **Bias Detection**: Test symmetry scores across ethnicities
2. **Language Safety**: Review all descriptions for neutrality
3. **Data Privacy**: No storage, GDPR compliance
4. **Misuse Prevention**: Rate limiting, no batch API, no embeddings export

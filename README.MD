# 🖼️ image-bulk-tinyfier

## A Nimble Solution for Mass Image Processing

This Python script tackles the surprisingly complex challenge of batch processing thousands of images while preserving
directory structures. With parallel processing capabilities and comprehensive logging, it transforms what would be days
of manual work into a streamlined, automated workflow.

## Core Functionality

At its essence, image-bulk-tinyfier delivers four crucial capabilities:

- Consumes a CSV file containing relative image paths
- Processes each image with customizable parameters (resize, optimize, format conversion)
- Faithfully recreates directory hierarchies at the destination
- Provides detailed analytics on processing outcomes and efficiency gains

What makes it particularly valuable is the balance between simplicity and power – a single-dependency tool that scales
from modest batches to enterprise-level image libraries.

## The Feedback Experience

What sets this tool apart is its communication layer. Rather than the traditional black-box approach of many processing
utilities, image-bulk-tinyfier keeps you informed with:

```
[=================>                   ] 42% 420/1000 | 35.2 img/s | 12m elapsed | ETA: 16m
```

After completion, you're presented with an actionable summary:

```
================================================================================
PROCESSING SUMMARY
================================================================================
Total images processed:     1000
Successfully processed:     998 (99.8%)
Failed:                     2 (0.2%)
Total processing time:      432.1 seconds
Average processing speed:   2.31 images/second

Original size:              1256.32 MB
Processed size:             312.57 MB
Storage saved:              943.75 MB (75.1%)

FAILED IMAGES:
--------------------------------------------------------------------------------
1. path/to/corrupted_image.jpg: Error processing path/to/corrupted_image.jpg: cannot identify image file
2. another/path/broken.png: Error processing another/path/broken.png: broken data stream when reading image file

Check the log file for complete details: processing_log.json
```

## Technical Requirements & Setup

- Python 3.8+
- Pillow library (the only dependency)

```bash
# Clone the repository
git clone https://github.com/poacosta/image-bulk-tinyfier.git
cd image-bulk-tinyfier

# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependency
pip install pillow
```

## Command Options

The tool offers a range of parameters to customize your processing pipeline:

| Option          | Purpose                        | Default             |
|-----------------|--------------------------------|---------------------|
| `--csv`         | CSV with image paths           | *Required*          |
| `--source`      | Original image location        | *Required*          |
| `--dest`        | Processed image destination    | *Required*          |
| `--workers`     | Parallel processing threads    | 8                   |
| `--max-width`   | Maximum width after resizing   | 600                 |
| `--max-height`  | Maximum height after resizing  | 600                 |
| `--quality`     | JPEG quality (0-100)           | 80                  |
| `--log-file`    | Processing log path            | processing_log.json |
| `--no-progress` | Disable progress visualization | False               |
| `--debug`       | Enable verbose logging         | False               |

## Performance Considerations

The tool's architecture is designed for flexibility across varied computing environments:

- **CPU Optimization**: Thread count automatically scales to your system's capabilities
- **Memory Efficiency**: Processes images in controlled batches to avoid memory pressure
- **Storage Awareness**: Optimized for both HDD and SSD configurations
- **Scale Testing**: Successfully deployed on datasets exceeding 400,000 images

## Technical Implementation

What might appear simple on the surface is backed by thoughtful engineering:

- Concurrent processing via ThreadPoolExecutor
- Comprehensive exception handling for robust operation
- Type-annotated codebase maintaining 9.5+ Pylint score
- Modular architecture with clear separation of concerns

The JSON log output provides a complete audit trail that can be parsed for integration with other tools or reporting
systems.

## CSV Input Format

The CSV format is intentionally minimalist – one relative path per line:

```
product/primary/item5562.jpg
marketing/banners/summer_promo.jpg
user/avatars/default.png
```

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*Built by someone who briefly contemplated manually resizing 400,000 images before coming to their senses. The
structured logging emerged from the realization that knowing what happened is often as important as making it happen.
Sometimes the most practical tools come from staring at a mountain of work and thinking "there must be a better way."*
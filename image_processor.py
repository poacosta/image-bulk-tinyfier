#!/usr/bin/env python3
"""
Image Bulk Tinyfier - Efficient batch image processing pipeline for large datasets.

This module provides functionality to process large collections of images,
resizing and optimizing them while preserving directory structure.
"""

import csv
import json
import logging
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

from PIL import Image  # Pillow library for image processing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_image(
        source_path: str,
        dest_path: str,
        dimensions: Tuple[int, int] = (600, 600),
        quality: int = 80
) -> Tuple[bool, Optional[str]]:
    """
    Process an image by resizing and optimizing it, then save to destination path.

    Args:
        source_path: Path to source image
        dest_path: Path where processed image should be saved
        dimensions: Maximum width and height
        quality: JPEG quality (0-100)

    Returns:
        Tuple of (success status, error message if any)
    """
    try:
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Open and process the image
        with Image.open(source_path) as img:
            original_size = img.size
            original_format = img.format

            # Convert to RGB if image has alpha channel (RGBA)
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # Resize the image maintaining aspect ratio
            img.thumbnail(dimensions, Image.Resampling.LANCZOS)
            new_size = img.size

            # Save with optimization
            img.save(
                dest_path,
                'JPEG',
                quality=quality,
                optimize=True,
                progressive=True
            )

            # Calculate size reduction
            if os.path.exists(source_path) and os.path.exists(dest_path):
                original_bytes = os.path.getsize(source_path)
                new_bytes = os.path.getsize(dest_path)
                reduction_pct = (1 - (new_bytes / original_bytes)) * 100 if original_bytes > 0 else 0
            else:
                original_bytes = 0
                new_bytes = 0
                reduction_pct = 0

            result_info = {
                "original_format": original_format,
                "original_size": original_size,
                "new_size": new_size,
                "original_bytes": original_bytes,
                "new_bytes": new_bytes,
                "reduction_pct": reduction_pct
            }

            logger.debug("Processed: %s -> %s", source_path, dest_path)
            return True, json.dumps(result_info)

    except (IOError, OSError, Image.UnidentifiedImageError) as e:
        error_msg = f"Error processing {source_path}: {str(e)}"
        logger.debug(error_msg)
        return False, error_msg


def update_progress(
        processed_count: int,
        total_img_count: int,
        start_time: float,
        width: int = 50,
        elapsed_update_interval: int = 1
) -> None:
    """
    Display progress bar and processing statistics.

    Args:
        processed_count: Current number of processed items
        total_img_count: Total number of items to process
        start_time: Processing start time
        width: Width of the progress bar in characters
        elapsed_update_interval: How often to update ETA (in seconds)
    """
    elapsed = time.time() - start_time
    progress = processed_count / total_img_count if total_img_count > 0 else 0
    bar_length = int(width * progress)

    # Only update ETAs periodically to avoid constant recalculation
    static_update = int(elapsed) % elapsed_update_interval == 0 or progress >= 1

    if progress > 0 and static_update:
        eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
        items_per_sec = processed_count / elapsed if elapsed > 0 else 0

        # Format the ETA nicely
        if eta >= 3600:
            eta_formatted = f"{int(eta / 3600)}h {int((eta % 3600) / 60)}m"
        elif eta >= 60:
            eta_formatted = f"{int(eta / 60)}m {int(eta % 60)}s"
        else:
            eta_formatted = f"{int(eta)}s"

        # Format elapsed time nicely
        if elapsed >= 3600:
            elapsed_formatted = f"{int(elapsed / 3600)}h {int((elapsed % 3600) / 60)}m"
        elif elapsed >= 60:
            elapsed_formatted = f"{int(elapsed / 60)}m {int(elapsed % 60)}s"
        else:
            elapsed_formatted = f"{int(elapsed)}s"

        # Create the progress bar visual
        progress_bar = f"[{'=' * bar_length}>{' ' * (width - bar_length)}]"

        # Create the stats line
        stats = (f"{processed_count}/{total_img_count} | {items_per_sec:.1f} img/s | "
                 f"{elapsed_formatted} elapsed | ETA: {eta_formatted}")

        # Print progress and stats
        sys.stdout.write(f"\r{progress_bar} {int(progress * 100)}% {stats}")
    else:
        # Just update the progress bar for intermediate updates
        progress_bar = f"[{'=' * bar_length}>{' ' * (width - bar_length)}]"
        sys.stdout.write(f"\r{progress_bar} {int(progress * 100)}%")

    sys.stdout.flush()

    # Add newline when complete
    if processed_count >= total_img_count:
        print()


def print_summary(results: Dict[str, Any], log_path: str) -> None:
    """
    Print processing summary to console.

    Args:
        results: Results dictionary with summary information
        log_path: Path to the log file
    """
    summary = results["summary"]

    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total images processed:     {summary['total_images']}")

    if summary['total_images'] > 0:
        success_pct = summary['successful'] / summary['total_images'] * 100
        fail_pct = summary['failed'] / summary['total_images'] * 100
    else:
        success_pct = 0
        fail_pct = 0

    print(f"Successfully processed:     {summary['successful']} ({success_pct:.1f}%)")
    print(f"Failed:                     {summary['failed']} ({fail_pct:.1f}%)")
    print(f"Total processing time:      {summary['processing_time']:.1f} seconds")

    # Calculate and print speed
    if summary['processing_time'] > 0:
        speed = summary['total_images'] / summary['processing_time']
        print(f"Average processing speed:   {speed:.2f} images/second")

    # Show storage stats
    original_mb = summary["total_bytes_original"] / (1024 * 1024)
    processed_mb = summary["total_bytes_processed"] / (1024 * 1024)
    saved_mb = original_mb - processed_mb
    print(f"Original size:              {original_mb:.2f} MB")
    print(f"Processed size:             {processed_mb:.2f} MB")
    print(f"Storage saved:              {saved_mb:.2f} MB ({summary['average_reduction']:.1f}%)")

    # Print information about failures if any
    if summary["failed"] > 0:
        print("\nFAILED IMAGES:")
        print("-" * 80)
        # Show up to 10 failures with ellipsis if there are more
        for i, failure in enumerate(results["failure"][:10]):
            print(f"{i + 1}. {failure['path']}: {failure['error']}")

        if len(results["failure"]) > 10:
            print(f"... and {len(results['failure']) - 10} more failures")

        print(f"\nCheck the log file for complete details: {log_path}")


def read_image_paths(
        csv_path: str,
        source_base_dir: str,
        dest_base_dir: str
) -> Tuple[List[Tuple[str, str, str]], int]:
    """
    Read image paths from CSV and prepare processing tasks.

    Args:
        csv_path: Path to CSV file with relative image paths
        source_base_dir: Base directory for source images
        dest_base_dir: Base directory where processed images will be saved

    Returns:
        Tuple of (list of tasks, total count)
    """
    tasks = []
    total_count = 0

    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # Skip header if present
        _ = next(reader, None)

        for row in reader:
            if not row:
                continue

            rel_path = row[0]  # Assuming the relative path is in the first column
            source_path = os.path.join(source_base_dir, rel_path)
            dest_path = os.path.join(dest_base_dir, rel_path)

            tasks.append((source_path, dest_path, rel_path))
            total_count += 1

    return tasks, total_count


def handle_processing_result(
        results: Dict[str, Any],
        rel_path: str,
        is_success: bool,
        message: str
) -> None:
    """
    Process and record the result of an image processing task.

    Args:
        results: Results dictionary to update
        rel_path: Relative path of the processed image
        is_success: Whether processing was successful
        message: Result message or error
    """
    timestamp = datetime.now().isoformat()

    if is_success:
        # Parse the result info from JSON
        try:
            result_info = json.loads(message)

            # Update summary stats
            results["summary"]["total_bytes_original"] += result_info.get("original_bytes", 0)
            results["summary"]["total_bytes_processed"] += result_info.get("new_bytes", 0)

            # Record success
            results["success"].append({
                "path": rel_path,
                "timestamp": timestamp,
                "original_format": result_info.get("original_format"),
                "original_size": result_info.get("original_size"),
                "new_size": result_info.get("new_size"),
                "original_bytes": result_info.get("original_bytes"),
                "new_bytes": result_info.get("new_bytes"),
                "reduction_pct": result_info.get("reduction_pct")
            })
        except (json.JSONDecodeError, TypeError):
            # If result parsing fails, just record basic success
            results["success"].append({
                "path": rel_path,
                "timestamp": timestamp,
                "message": "Success, but details unavailable"
            })
    else:
        # Record failure
        results["failure"].append({
            "path": rel_path,
            "timestamp": timestamp,
            "error": message
        })


def process_images_from_csv(
        csv_path: str,
        source_base_dir: str,
        dest_base_dir: str,
        max_workers: int = 8,
        img_dimensions: Tuple[int, int] = (600, 600),
        quality: int = 80,
        show_progress: bool = True,
        log_path: Optional[str] = None
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Process images listed in a CSV file with paths relative to source_base_dir.

    Args:
        csv_path: Path to CSV file with relative image paths
        source_base_dir: Base directory for source images
        dest_base_dir: Base directory where processed images will be saved
        max_workers: Maximum number of worker threads
        img_dimensions: Maximum dimensions (width, height) for resized images
        quality: JPEG quality (0-100)
        show_progress: Whether to display a progress bar
        log_path: Path to save processing logs

    Returns:
        Tuple[int, int, Dict]: (successful count, total count, results dictionary)
    """
    successful_count = 0
    start_time = time.time()

    # Initialize results tracking
    results = {
        "success": [],
        "failure": [],
        "summary": {
            "start_time": datetime.now().isoformat(),
            "total_images": 0,
            "successful": 0,
            "failed": 0,
            "total_bytes_original": 0,
            "total_bytes_processed": 0,
            "processing_time": 0,
            "average_reduction": 0,
        }
    }

    # Create destination directory if it doesn't exist
    os.makedirs(dest_base_dir, exist_ok=True)

    # Read image paths and prepare tasks
    tasks, total_count = read_image_paths(csv_path, source_base_dir, dest_base_dir)

    if show_progress:
        logger.info("Found %s files to process", total_count)
        logger.info("Processing %s images with %s workers...", total_count, max_workers)
        update_progress(0, total_count, start_time)

    # Process all tasks with progress tracking
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_image, source_path, dest_path, img_dimensions, quality):
                (source_path, dest_path, rel_path)
            for source_path, dest_path, rel_path in tasks
        }

        completed = 0
        for future in as_completed(futures):
            _, _, rel_path = futures[future]
            is_success, message = future.result()

            completed += 1

            if is_success:
                successful_count += 1

            # Record result
            handle_processing_result(results, rel_path, is_success, message)

            if show_progress:
                update_progress(completed, total_count, start_time)

    # Complete summary data
    processing_time = time.time() - start_time
    results["summary"]["end_time"] = datetime.now().isoformat()
    results["summary"]["processing_time"] = processing_time
    results["summary"]["total_images"] = total_count
    results["summary"]["successful"] = successful_count
    results["summary"]["failed"] = total_count - successful_count

    # Calculate average reduction percentage
    if results["summary"]["total_bytes_original"] > 0:
        savings = 1 - (results["summary"]["total_bytes_processed"] / results["summary"]["total_bytes_original"])
        results["summary"]["average_reduction"] = savings * 100
    else:
        results["summary"]["average_reduction"] = 0

    # Print summary to console
    if show_progress:
        print_summary(results, log_path or "")

    # Save results to log file if requested
    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        logger.info("Processing log saved to: %s", log_path)

    return successful_count, total_count, results


def main() -> None:
    """Main function to parse arguments and process images."""

    parser = argparse.ArgumentParser(description='Process images based on a CSV file.')
    parser.add_argument('--csv', required=True, help='Path to CSV file with image paths')
    parser.add_argument('--source', required=True, help='Base directory for source images')
    parser.add_argument('--dest', required=True, help='Base directory for processed images')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads')
    parser.add_argument('--max-width', type=int, default=600,
                        help='Maximum width for resized images')
    parser.add_argument('--max-height', type=int, default=600,
                        help='Maximum height for resized images')
    parser.add_argument('--quality', type=int, default=80,
                        help='JPEG quality (0-100)')
    parser.add_argument('--log-file', type=str, default='processing_log.json',
                        help='Path to save processing log (JSON format)')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bar')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Starting image processing from %s", args.csv)
    logger.info("Source directory: %s", args.source)
    logger.info("Destination directory: %s", args.dest)
    logger.info("Target dimensions: %sx%s, Quality: %s%%",
                args.max_width, args.max_height, args.quality)

    target_dimensions = (args.max_width, args.max_height)

    process_images_from_csv(
        args.csv,
        args.source,
        args.dest,
        args.workers,
        target_dimensions,
        args.quality,
        not args.no_progress,
        args.log_file
    )


if __name__ == "__main__":
    main()

"""
Cricket Biomechanics Analysis - Main Entry Point

This script provides a command-line interface to the cricket biomechanics
analysis system. It allows users to analyze cricket straight drive videos,
extracting pose data, calculating biomechanical features, and providing
technique feedback.

Usage:
    python main.py --video PATH_TO_VIDEO
    python main.py --batch PATH_TO_DIRECTORY
    python main.py --help

Example:
    python main.py --video data/raw/cricket_drive.mp4
    python main.py --batch data/raw
"""

import os
import sys
import argparse
import time
from pathlib import Path

from src.inference.cricket_analyzer import CricketTechniqueAnalyzer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cricket Biomechanics Analysis System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str, help='Path to video file for analysis')
    group.add_argument('--batch', type=str, help='Path to directory containing videos for batch analysis')
    
    # Optional arguments
    parser.add_argument('--output', type=str, help='Output directory for analysis results')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--no-vis', action='store_true', 
                        help='Disable visualization outputs')
    parser.add_argument('--extensions', type=str, default='.mp4,.avi,.mov',
                        help='Comma-separated list of video file extensions for batch processing')
    
    return parser.parse_args()


def main():
    """Main function to run the cricket technique analysis."""
    args = parse_args()
    
    # Ensure the config path is resolved
    config_path = os.path.abspath(args.config)
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return 1
    
    # Initialize the analyzer
    analyzer = CricketTechniqueAnalyzer(
        config_path=config_path,
        model_path=args.model
    )
    
    # Set up output directory
    output_dir = args.output
    if not output_dir:
        if args.video:
            output_dir = os.path.join(os.path.dirname(args.video), 'analysis')
        elif args.batch:
            output_dir = os.path.join(args.batch, 'analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process according to the provided arguments
    start_time = time.time()
    
    try:
        if args.video:
            # Process a single video
            video_path = os.path.abspath(args.video)
            
            if not os.path.exists(video_path):
                print(f"Error: Video file not found at {video_path}")
                return 1
            
            print(f"Processing video: {video_path}")
            results = analyzer.analyze_video(
                video_path,
                output_dir=output_dir,
                visualize=not args.no_vis
            )
            
            # Print brief summary
            print("\nAnalysis Summary:")
            print(f"- Video: {results['video_name']}")
            print(f"- Total frames: {results['total_frames']}")
            print(f"- Processing time: {results['duration']:.2f} seconds")
            print(f"- Phase breakdown: {results['phase_breakdown']}")
            print(f"- Results saved to: {output_dir}")
            
        elif args.batch:
            # Process a batch of videos
            batch_dir = os.path.abspath(args.batch)
            
            if not os.path.exists(batch_dir) or not os.path.isdir(batch_dir):
                print(f"Error: Batch directory not found at {batch_dir}")
                return 1
            
            # Parse extensions
            extensions = args.extensions.split(',')
            
            print(f"Processing all videos in: {batch_dir}")
            results = analyzer.analyze_batch(
                batch_dir,
                output_dir=output_dir,
                file_extensions=extensions
            )
            
            # Print brief summary
            print("\nBatch Analysis Summary:")
            print(f"- Videos processed: {len(results)}")
            print(f"- Total processing time: {time.time() - start_time:.2f} seconds")
            print(f"- Results saved to: {output_dir}")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Prepare MIDI data for training by extracting from Maestro dataset.

This script automatically:
1. Finds the Maestro dataset
2. Lists available composers
3. Extracts MIDI files for the selected composer
4. Saves them to data/midi/{composer_name}/ automatically
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from performance_rnn_torch.utils import paths
from performance_rnn_torch.utils.logger import setup_logger

logger = setup_logger('prepare_data', file=False)


def find_maestro_dataset() -> Optional[Path]:
    """
    Find the Maestro dataset directory.

    Searches in:
    1. data/maestro-v3.0.0/ (standard location)
    2. Current directory

    Returns:
        Path to maestro directory, or None if not found
    """
    project_root = paths.project_root

    possible_locations = [
        project_root / "data" / "maestro-v3.0.0",
        Path.cwd() / "maestro-v3.0.0",
    ]

    for location in possible_locations:
        csv_file = location / "maestro-v3.0.0.csv"
        if csv_file.exists():
            logger.info(f"Found Maestro dataset at: {location}")
            return location

    return None


def get_available_composers(maestro_dir: Path) -> List[str]:
    """
    Get list of available composers from Maestro dataset.

    Args:
        maestro_dir: Path to maestro dataset directory

    Returns:
        Sorted list of composer names
    """
    csv_file = maestro_dir / "maestro-v3.0.0.csv"
    df = pd.read_csv(csv_file)
    composers = sorted(df['canonical_composer'].unique())
    return composers


def list_composers(maestro_dir: Path):
    """Print available composers in Maestro dataset."""
    composers = get_available_composers(maestro_dir)

    logger.info("\n" + "="*70)
    logger.info(f"Available composers in Maestro dataset ({len(composers)} total):")
    logger.info("="*70)

    for i, composer in enumerate(composers, 1):
        csv_file = maestro_dir / "maestro-v3.0.0.csv"
        df = pd.read_csv(csv_file)
        count = len(df[df['canonical_composer'] == composer])
        logger.info(f"{i:2d}. {composer:30s} ({count:3d} pieces)")

    logger.info("="*70 + "\n")


def extract_composer_midis(
    composer: str,
    maestro_dir: Path,
    output_dir: Optional[Path] = None
) -> int:
    """
    Extract MIDI files for a specific composer.

    Args:
        composer: Name of composer (must match exactly)
        maestro_dir: Path to maestro dataset directory
        output_dir: Custom output directory (default: data/midi/{composer}/

    Returns:
        Number of files copied
    """
    csv_file = maestro_dir / "maestro-v3.0.0.csv"
    df = pd.read_csv(csv_file)

    # Check if composer exists
    available_composers = get_available_composers(maestro_dir)
    if composer not in available_composers:
        logger.error(f"\nComposer '{composer}' not found!")
        logger.error("Did you mean one of these?")
        # Find close matches
        close_matches = [c for c in available_composers if composer.lower() in c.lower()]
        if close_matches:
            for match in close_matches[:5]:
                logger.error(f"  - {match}")
        else:
            logger.error("Run without --composer to see all available composers")
        return 0

    # Get MIDI files for this composer
    composer_files = df[df['canonical_composer'] == composer]['midi_filename'].tolist()

    if not composer_files:
        logger.error(f"No MIDI files found for composer: {composer}")
        return 0

    # Determine output directory
    if output_dir is None:
        # Sanitize composer name for directory
        safe_composer_name = composer.lower().replace(' ', '_').replace('.', '')
        output_dir = paths.midi_dir / safe_composer_name

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nExtracting {len(composer_files)} MIDI files for: {composer}")
    logger.info(f"Output directory: {output_dir}\n")

    # Copy files
    copied_count = 0
    failed_count = 0

    for midi_file in composer_files:
        source_path = maestro_dir / midi_file

        if not source_path.exists():
            logger.warning(f"File not found: {source_path}")
            failed_count += 1
            continue

        # Use just the filename (not the full path with year)
        dest_path = output_dir / source_path.name

        try:
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            if copied_count % 10 == 0:
                logger.info(f"Copied {copied_count}/{len(composer_files)} files...")
        except Exception as e:
            logger.error(f"Failed to copy {source_path.name}: {e}")
            failed_count += 1

    logger.info(f"\n✓ Successfully copied {copied_count} files to: {output_dir}")
    if failed_count > 0:
        logger.warning(f"✗ Failed to copy {failed_count} files")

    return copied_count


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Extract MIDI files from Maestro dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available composers
  python scripts/prepare_data.py --list

  # Extract all pieces by Claude Debussy
  python scripts/prepare_data.py --composer "Claude Debussy"

  # Extract to custom directory
  python scripts/prepare_data.py --composer "Franz Schubert" --output custom_dir/

  # Specify maestro location
  python scripts/prepare_data.py --composer "Bach" --maestro-dir /path/to/maestro
        """
    )

    parser.add_argument(
        '-c', '--composer',
        type=str,
        default=None,
        help='Composer name (must match exactly). Use --list to see available composers.'
    )

    parser.add_argument(
        '-l', '--list',
        action='store_true',
        help='List all available composers in Maestro dataset'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help=f'Output directory (default: data/midi/{{composer_name}}/)'
    )

    parser.add_argument(
        '-m', '--maestro-dir',
        type=str,
        default=None,
        help='Path to maestro-v3.0.0 directory (auto-detected if not specified)'
    )

    return parser.parse_args()


def main(args=None):
    if args is None:
        args = get_arguments()

    # Find Maestro dataset
    if args.maestro_dir:
        maestro_dir = Path(args.maestro_dir)
        if not maestro_dir.exists():
            logger.error(f"Maestro directory not found: {maestro_dir}")
            return 1
    else:
        maestro_dir = find_maestro_dataset()
        if maestro_dir is None:
            logger.error("\n" + "="*70)
            logger.error("Maestro dataset not found!")
            logger.error("="*70)
            logger.error("\nPlease download the Maestro dataset v3.0.0:")
            logger.error("  https://magenta.tensorflow.org/datasets/maestro")
            logger.error("\nExtract it to:")
            logger.error("  data/maestro-v3.0.0/")
            logger.error("\nOr specify a custom location with:")
            logger.error("  --maestro-dir /path/to/maestro-v3.0.0")
            logger.error("="*70 + "\n")
            return 1

    # List composers if requested
    if args.list:
        list_composers(maestro_dir)
        return 0

    # Extract composer files
    if args.composer:
        output_dir = Path(args.output) if args.output else None
        count = extract_composer_midis(args.composer, maestro_dir, output_dir)

        if count > 0:
            logger.info("\n" + "="*70)
            logger.info("Next steps:")
            logger.info("="*70)
            logger.info("1. Preprocess the MIDI files:")
            logger.info("   python scripts/preprocess.py")
            logger.info("")
            logger.info("2. Train the model:")
            logger.info("   python scripts/train.py")
            logger.info("="*70 + "\n")

        return 0 if count > 0 else 1

    # No action specified
    logger.info("No action specified. Use --list to see composers or --composer to extract.")
    logger.info("Run with --help for more information.")
    return 1


if __name__ == '__main__':
    sys.exit(main())

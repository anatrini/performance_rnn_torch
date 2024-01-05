import argparse
import os
import pandas as pd
import shutil
import sys


def copy_select(composer, output_dir, csv_filepath):

    df = pd.read_csv(csv_filepath)
    selected = df[df['canonical_composer'] == composer]['midi_filename'].apply(os.path.basename)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        print(f"Permission denied: '{output_dir}'")
        return

    for file in selected:
        found = False
        for dirpath, _, filenames in os.walk('.'):
            if file in filenames:
                try:
                    shutil.copy(os.path.join(dirpath, file), output_dir)
                    found = True
                except IOError as e:
                    print(f"Unable to copy file: {e}")
                except:
                    print("Unexpected error:", sys.exc_info())
        if not found:
            print(f"File not found: '{file}'")


def main():
    parser = argparse.ArgumentParser(description='Copy midi files from the Maestro Dataset according to the composer name.')
    parser.add_argument('composer', help='Composer\'s name')
    parser.add_argument('output_dir', help='The directory you want to copy midi files to.')
    parser.add_argument('csv_filepath', help='The Mestro\'s .csv dataset content')

    args = parser.parse_args()

    copy_select(args.composer, args.output_dir, args.csv_filepath)


if __name__ == '__main__':
    main()
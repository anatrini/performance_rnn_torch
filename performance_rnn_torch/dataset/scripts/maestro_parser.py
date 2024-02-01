import optparse
import os
import pandas as pd
import shutil
import sys


def get_options():
    parser = optparse.OptionParser()

    parser.add_option('-c', '--composer',
                      dest='composer',
                      type='string',
                      default=None,
                      help='Name of the composer you want to select midi files of')
    
    parser.add_option('-s', '--output_dir',
                      dest='output_dir',
                      default=None,
                      help='Name of the directory you want to save selected midi files to')
    
    parser.add_option('-m', '--maestro_file',
                      dest='maestro_file',
                      default=None,
                      help='The path to the .csv Maestro\'s dataset content')
    
    return parser.parse_args()[0]



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
                    print(f'File copied succesfully to {output_dir}')
                    found = True
                except IOError as e:
                    print(f"Unable to copy file: {e}")
                except:
                    print("Unexpected error:", sys.exc_info())
        if not found:
            print(f"File not found: '{file}'")


def main():
    options = get_options()
    composer = options.composer
    output_dir = options.output_dir
    maestro_file = options.maestro_file

    copy_select(composer, output_dir, maestro_file)



if __name__ == '__main__':
    main()
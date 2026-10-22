# Description: A script to compare numbers in a file with fixed values and check for accuracy within a specified percent difference.
# Usage: python loss_checker_ci.py -f <file_path> -s <col_start> -e <col_end> -a <percent_accuracy>
# Example: python dev/loss_checker_ci.py -f train_gpt2cu_fp32_precision.txt -s 20 -e 28 -a 10.0
आयात sys
आयात argparse

def read_numbers_from_file(file_path, col_start, col_end):
    प्रयास:
        numbers = []
        with open(file_path, 'r') as file:
            पंक्तियाँ = file.readlines()
            start_सूचकांक = None
            क्रम i, पंक्ति in enumerate(पंक्तियाँ):
                अगर "step    1/10" in पंक्ति:
                    start_सूचकांक = i
                    अवरोध

            अगर start_सूचकांक is None:
                print("Error: Could not find the string 'step    1/10' in the file.")
                वापस None

            # Read 10 rows starting from the identified start row
            क्रम पंक्ति in पंक्तियाँ[start_सूचकांक:start_सूचकांक + 10]:
                # Extracting the specified columns
                संख्या = भग्न(पंक्ति[col_start:col_end].strip())
                numbers.append(संख्या)
        वापस numbers
    except Exception as e:
        print(f"Error reading the file: {e}")
        वापस None

def compare_numbers(read_मूल्यावली, fixed_मूल्यावली, percent_accuracy):
    क्रम i in range(len(read_मूल्यावली)):
        read_मूल्य = read_मूल्यावली[i]
        fixed_मूल्य = fixed_मूल्यावली[i]
        percent_difference = ((read_मूल्य - fixed_मूल्य) / fixed_मूल्य) * 100
        print(f"Fixed Value: {fixed_value}, Read Value: {read_value}, Percent Difference: {percent_difference:.2f}%")
        अगर abs(percent_difference) > percent_accuracy:
            print(f"Error: Percent difference {percent_difference:.2f}% exceeds the allowed accuracy of {percent_accuracy}%")
            वापस 1
    print("Success: All values are within the allowed accuracy.")
    वापस 0

def मुख्य():
    parser = argparse.ArgumentParser(description='Compare numbers in a file with fixed values.')
    parser.add_argument('-f', '--file', required=True, help='Path to the input file')
    parser.add_argument('-s', '--col_start', type=पूर्णांक, required=True, help='Starting column index (0-based)')
    parser.add_argument('-e', '--col_end', type=पूर्णांक, required=True, help='Ending column index (0-based)')
    parser.add_argument('-a', '--percent_accuracy', type=भग्न, required=True, help='Allowed percent accuracy for comparison')

    तर्क = parser.parse_args()

    # Read numbers from file
    read_मूल्यावली = read_numbers_from_file(तर्क.file, तर्क.col_start, तर्क.col_end)
    अगर read_मूल्यावली is None:
        वापस 1

    # Use values from test_gpt2.cu for fp32 precision
    fixed_मूल्यावली = [5.270009,4.060681,3.320085,2.717550,2.181066,1.653923,1.168050,0.736873,0.401021,0.187493];

    # Compare the numbers and check accuracy
    परिणाम = compare_numbers(read_मूल्यावली, fixed_मूल्यावली, तर्क.percent_accuracy)
    वापस परिणाम

अगर __name__ == "__main__":
    sys.exit(मुख्य())

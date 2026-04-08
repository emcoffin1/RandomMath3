import csv

input_file = "nozzle_curve.csv"
output_file = "nozzle_curve(1).csv"

with open(input_file, newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Take only the first two columns
        writer.writerow(row[:2])
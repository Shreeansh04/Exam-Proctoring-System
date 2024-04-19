import csv

def load_ranges(file_path):
    ranges = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            start = int(float(row[0]) * 1000)  # Convert seconds to milliseconds
            end = int(float(row[1]) * 1000)
            ranges.append((start, end))
    return ranges

def generate_cheating_csv(ranges, end_time):
    cheating_data = []
    timestamp = 0

    while timestamp <= end_time:
        cheating = 0
        for start, end in ranges:
            if start <= timestamp <= end:
                cheating = 1
                break

        cheating_data.append([timestamp, cheating])
        timestamp += 40

    # Merge overlapping ranges
    merged_data = []
    prev_timestamp, prev_cheating = cheating_data[0]
    for timestamp, cheating in cheating_data[1:]:
        if cheating == prev_cheating:
            continue
        else:
            if prev_cheating == 1:
                merged_data.append((prev_timestamp, timestamp))
            prev_timestamp, prev_cheating = timestamp, cheating

    # Convert merged ranges to individual rows
    final_data = []
    for start, end in merged_data:
        final_data.append([start, 1])
        while start + 40 < end:
            start += 40
            final_data.append([start, 1])
        final_data.append([end, 1])

    # Fill in remaining timestamps with 0
    timestamp = 0
    i = 0
    while timestamp <= end_time:
        if i < len(final_data) and timestamp == final_data[i][0]:
            final_data[i].append(timestamp)
            i += 1
        else:
            final_data.append([0, timestamp])
        timestamp += 40

    return final_data

# Load actual ranges from the CSV file
actual_ranges = load_ranges('actual_ranges.csv')

# Generate cheating data
cheating_data = generate_cheating_csv(actual_ranges, 1143000)

# Write cheating data to a new CSV file
with open('cheating.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['cheating', 'timestamp'])
    writer.writerows(cheating_data)
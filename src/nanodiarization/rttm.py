
def convert_rttm(input_text:str):
    lines = input_text.strip().split('\n')
    speaker_map = {}
    current_speaker = 'A'
    segments = []

    for line in lines:
        parts = line.split()
        start_time = float(parts[3])
        duration = float(parts[4])
        speaker_label = parts[7]
        segments.append([start_time, duration, speaker_label])

    segments = sorted(segments, key=lambda x: x[0])

    for seg in segments:
        if seg[-1] not in speaker_map:
            speaker_map[seg[-1]] = current_speaker
            current_speaker = chr(ord(current_speaker) + 1)
        seg[-1] = speaker_map[seg[-1]]

    return segments

if __name__ == "__main__":
    example="""
    SPEAKER azisu 1 7.960000 5.720000 <NA> <NA> spk00 <NA> <NA>
    SPEAKER azisu 1 138.600000 13.840000 <NA> <NA> spk00 <NA> <NA>
    SPEAKER azisu 1 156.880000 6.480000 <NA> <NA> spk00 <NA> <NA>
    SPEAKER azisu 1 169.840000 4.920000 <NA> <NA> spk00 <NA> <NA>
    SPEAKER azisu 1 180.160000 1.960000 <NA> <NA> spk00 <NA> <NA>
    SPEAKER azisu 1 95.440000 1.000000 <NA> <NA> spk01 <NA> <NA>
    SPEAKER azisu 1 102.000000 36.600000 <NA> <NA> spk01 <NA> <NA>
    SPEAKER azisu 1 150.760000 7.360000 <NA> <NA> spk01 <NA> <NA>
    SPEAKER azisu 1 159.160000 12.560000 <NA> <NA> spk01 <NA> <NA>
    SPEAKER azisu 1 172.480000 2.800000 <NA> <NA> spk01 <NA> <NA>
    SPEAKER azisu 1 51.200000 2.240000 <NA> <NA> spk02 <NA> <NA>
    SPEAKER azisu 1 36.960000 7.360000 <NA> <NA> spk00 <NA> <NA>
    SPEAKER azisu 1 74.160000 0.520000 <NA> <NA> spk03 <NA> <NA>
    SPEAKER azisu 1 174.480000 3.240000 <NA> <NA> spk03 <NA> <NA>
    SPEAKER azisu 1 177.880000 16.240000 <NA> <NA> spk03 <NA> <NA>
    SPEAKER azisu 1 125.200000 0.440000 <NA> <NA> spk00 <NA> <NA>
    """
    sorted_segments = convert_rttm(example)
    print("\n".join([str(x) for x in sorted_segments]))

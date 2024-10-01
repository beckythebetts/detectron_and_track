import detectron_infer
import track
import threshold_epi
import batch_extract_features

def main():
    cellpose_segment.main()
    track.main()
    threshold_epi.main()
    batch_extract_features.main()
    track_phagocytosis_events.main()

if __name__ == '__main__':
    main()
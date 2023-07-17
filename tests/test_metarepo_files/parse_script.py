from ms3 import Parse
from ms3.logger import get_logger


def main():
    p = Parse("~/unittest_metacorpus/mixed_files")
    p.parse_scores()
    t = get_logger("ms3.Parse.mixed_files.Did03M-Son_regina-1762-Sarti.mscx")
    filt = t.filters[0]
    print("IGNORED_WARNINGS")
    print(filt.ignored_warnings)
    t.warning(f"This should be a DEBUG message.", extra={"message_id": (2, 94)})
    _ = p.get_dataframes(expanded=True)

if __name__ == '__main__':
    main()
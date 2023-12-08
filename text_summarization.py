"""
    Extractive text summarization
"""
# Local import
import preprocessing
import utils
import page_rank

TRAIN_DIR = "DUC_TEXT/train"

def main():
    """
        Main function
    """
    accuracy = {}
    for file in preprocessing.get_all_files(TRAIN_DIR):
        file_path = TRAIN_DIR + "/" + file

        sentences = preprocessing.read_file(file_path)
        clean_data = preprocessing.get_clean_data(file_path)

        matrix = utils.get_similarity_matrix(clean_data)

        pr = page_rank.PageRank(numpy_array=matrix)
        rank = pr.get_page_rank()

        num_of_sum = utils.get_num_of_sent_in_summary(file)
        if num_of_sum == 0:
            num_of_sum = 10

        text_sum = utils.get_text_summarization(
            page_rank=rank,
            sentences=sentences,
            num_sum=num_of_sum
        )

        utils.write_file(contents=text_sum, file_name=file)

        expected_result = preprocessing.read_file(file_path="DUC_SUM/" + file)
        sum_sents = preprocessing.read_file(file_path="OUTPUT/" + file)
        accuracy[file] = utils.get_accuracy(expected_result, sum_sents)

    utils.write_file(contents=accuracy, file_name='Accuracy.json')


if __name__ == "__main__":
    main()

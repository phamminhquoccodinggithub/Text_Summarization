"""
    Extractive text summarization
"""
# Local import
import preprocessing
import utils
import page_rank
import constants as const


def main():
    """
        Main function
    """
    accuracy = {}

    files = preprocessing.get_all_files(const.TRAIN_DIR)
    for file in preprocessing.get_all_files(const.TEST_DIR):
        files.append(file)

    for file in files:
        if file not in preprocessing.get_all_files(const.TEST_DIR):
            file_path = const.TRAIN_DIR + file
        else:
            file_path = const.TEST_DIR + file

        sentences = preprocessing.read_file(file_path)
        clean_data = preprocessing.get_clean_data(file_path)
        matrix = utils.get_similarity_matrix(clean_data, technique=0)
        # vector = utils.tf_idf(clean_data)
        # vector = utils.word_to_vec(clean_data)
        # matrix = utils.get_similarity_matrix(sentences=vector, technique=1)
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

        utils.write_file(contents=text_sum, file_name=file, sum_dir=const.TEST_BOW_DIR)

        expected_result = preprocessing.read_file(file_path=const.EXPECTED_SUM_DIR + file)
        sum_sents = preprocessing.read_file(file_path=const.TEST_BOW_DIR + file)
        accuracy[file] = utils.get_accuracy(expected_result, sum_sents)

    utils.write_file(contents=accuracy, file_name='Accuracy.json', sum_dir=const.TEST_BOW_DIR)


if __name__ == "__main__":
    main()

import os
import youtube_dl
import nltk
nltk.download('wordnet')
import gensim
import speech_recognition as sr
from pydub import AudioSegment
from nltk.corpus import wordnet as wn

DATA_DIR = 'data'
PODCAST_FILE_NAME_PREFIX = 'podcast'
PODCAST_FILE_FORMAT = 'wav'
TEMP_FILE_NAME_PREFIX = 'tmp'
STEP = 100
ANALYZED_PERCENT = 0.15
NUM_TOPICS = 1
NUM_WORDS = 3


def main():
    podcast_file_name = download_podcast(
        url='https://www.youtube.com/watch?v=in7tepc2shg'
        # url='https://www.youtube.com/watch?v=J5O5iRpLXKY&t=637s'
    )
    text = recognize_text(podcast_file_name)
    tokens = prepare_text_for_lda(text)
    topics = get_topics_by_lda(tokens)
    [print(topic) for topic in topics]


def _get_file_name(file_name_prefix, file_format):
    return '{}.{}'.format(file_name_prefix, file_format)


def download_podcast(url=None):
    if url is not None:
        podcast_file_name = _get_file_name(PODCAST_FILE_NAME_PREFIX, PODCAST_FILE_FORMAT)
        # ydl_opts = {
        #     'format': 'bestaudio/best',
        #     'postprocessors': [{
        #         'key': 'FFmpegExtractAudio',
        #         'preferredcodec': PODCAST_FILE_FORMAT,
        #         'preferredquality': '192'
        #     }],
        #     'outtmpl': PODCAST_FILE_NAME_PREFIX,
        #     'prefer_ffmpeg': True,
        #     'keepvideo': False
        # }
        # with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        #     ydl.download([url])
        # os.rename(PODCAST_FILE_FORMAT, os.path.join(DATA_DIR, podcast_file_name))
        return podcast_file_name
    else:
        raise Exception('url is empty')


def recognize_text(podcast_file_name):
    text_chunks = []
    tmp_file_name = _get_file_name(TEMP_FILE_NAME_PREFIX, PODCAST_FILE_FORMAT)

    # AudioSegment.converter = which("ffmpeg")
    audio = AudioSegment.from_wav(os.path.join(DATA_DIR, podcast_file_name))
    fragments_count = int(len(audio) / (STEP * 1000) * ANALYZED_PERCENT)
    for i in range(fragments_count):
        print(i, '/', fragments_count)
        try:
            t1 = i * STEP * 1000  # Works in milliseconds
            t2 = (i+1) * STEP * 1000
            new_audio = audio[t1:t2]
            new_audio.export(os.path.join(DATA_DIR, tmp_file_name), format=PODCAST_FILE_FORMAT)

            recognizer = sr.Recognizer()
            with sr.AudioFile(os.path.join(DATA_DIR, tmp_file_name)) as source:
                podcast_audio = recognizer.record(source)
            text_chunks.append(recognizer.recognize_google(podcast_audio, language='ru_RU').lower())
        except Exception as e:
            print(e)
    text = ' '.join(text_chunks)
    print(text)
    with open(os.path.join(DATA_DIR, _get_file_name(PODCAST_FILE_NAME_PREFIX, 'txt1')), 'w+') as f:
        f.write(text)
    return text


def _tokenize_text(text):
    return text.split()


def _get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def prepare_text_for_lda(text):
    tokens = _tokenize_text(text)
    en_stop = set(nltk.corpus.stopwords.words('russian'))
    return [_get_lemma(token) for token in tokens
            if len(token) > 4 and token not in en_stop]


def get_topics_by_lda(tokens):
    dictionary = gensim.corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(text) for text in [tokens]]
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    return lda_model.print_topics(num_words=NUM_WORDS)


if __name__ == '__main__':
    main()

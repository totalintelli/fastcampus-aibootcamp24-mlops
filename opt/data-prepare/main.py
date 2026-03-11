'''
파일 명칭 : main.py
기     능 : 크롤러(crawler.py)와 전처리 모듈(preprocessing.py)을 호출
입     력 : 없음
출     력 : 없음
작  성 자 : 송 용 단
작성 일자 : 2026-03-11
'''
import pandas as pd
from dotenv import load_dotenv

from preprocessing import TMDBPreProcessor
from crawler import TMDBCrawler

load_dotenv()


def run_popular_movie_crawler():
    tmdb_crawler = TMDBCrawler()
    result = tmdb_crawler.get_bulk_popular_movies(start_page=1, end_page=1)
    tmdb_crawler.save_movies_to_json_file(result, "./result", "popular")

    tmdb_preprocessor = TMDBPreProcessor(result)
    tmdb_preprocessor.run()
    tmdb_preprocessor.save("watch_log")


if __name__ == '__main__':
    run_popular_movie_crawler()

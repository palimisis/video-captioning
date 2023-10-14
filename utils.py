import numpy as np
import urllib.request


def download_video(video_url: str) -> np.array:
    """
    Download video from url.
    :param video_url: url
    :return: video
    """
    urllib.request.urlretrieve(video_url, "video.mp4")


def download_videos(video_urls: np.array) -> np.array:
    """
    Download videos from a list of urls.
    :param video_urls: list of urls
    :return: list of videos
    """
    videos = []
    for video_url in video_urls:
        videos.append(download_video(video_url))
    return np.array(videos)

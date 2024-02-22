import os
import datetime
import uuid
from typing import List

import supervision as sv


MAX_VIDEO_LENGTH_SEC = 2


def generate_file_name(extension="mp4"):
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = uuid.uuid4()
    return f"{current_datetime}_{unique_id}.{extension}"


def list_files_older_than(directory: str, diff_minutes: int) -> List[str]:
    diff_seconds = diff_minutes * 60
    now = datetime.datetime.now()
    older_files: List[str] = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_mod_time = os.path.getmtime(file_path)
            file_mod_datetime = datetime.datetime.fromtimestamp(file_mod_time)
            time_diff = now - file_mod_datetime
            if time_diff.total_seconds() > diff_seconds:
                older_files.append(file_path)

    return older_files


def remove_files_older_than(directory: str, diff_minutes: int) -> None:
    older_files = list_files_older_than(directory, diff_minutes)
    file_count = len(older_files)

    for file_path in older_files:
        os.remove(file_path)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{now}] Removed {file_count} files older than {diff_minutes} minutes from "
        f"'{directory}' directory."
    )


def calculate_end_frame_index(source_video_path: str) -> int:
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    return min(
        video_info.total_frames,
        video_info.fps * MAX_VIDEO_LENGTH_SEC
    )


def create_directory(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

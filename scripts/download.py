# Need to install pytube to work `pip3 install pytube3`
from pytube import YouTube

# zapin_videos_path = './videos/zapin'

apsara_videos_path = './videos/predictions/apsara'
apsara_vid_links = [
    # 'https://www.youtube.com/watch?v=LbhoH4UZjd8',
    # 'https://www.youtube.com/watch?v=JuFd8KlmrJo',
    # 'https://www.youtube.com/watch?v=VZ7P_HbrB7M',
    # 'https://www.youtube.com/watch?v=qHRBig0ThM8',
    # 'https://www.youtube.com/watch?v=8Gw_1n0J_fI'
]

# tarian_jorget_videos_path = './videos/tarian_jorget'
# tarian_jorget_vid_links = [
#     'https://www.youtube.com/watch?v=NZX1qWOhr2M'
#     'https://www.youtube.com/watch?v=5u75EgS-Zsw',
#     'https://www.youtube.com/watch?v=ebtuTs1S2_w',
#     'https://www.youtube.com/watch?v=9Q89T2RUXmE',
#     'https://www.youtube.com/watch?v=ZWGRskGQR6E'

# ]

# khen_video_path = './videos/khen_2'
# khen_vid_links = [
    # 'https://www.youtube.com/watch?v=mGaIQma2jQc',
    # 'https://www.youtube.com/watch?v=LByVX7yU5gs',
    # 'https://www.youtube.com/watch?v=jfPeSrmHJKg',
    # 'https://www.youtube.com/watch?v=3bv3PN4t_UY',
#     'https://www.youtube.com/watch?v=9tUoWgwiLNY'
# ]

for link in apsara_vid_links:
    print("Downloading... %s" %(link))
    YouTube(link).streams.first().download(apsara_videos_path)

print('Done.')

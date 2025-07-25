# mymodel.py
class LyricsCommentData:
    def __init__(self, music4all_id=None, songmeanings_id=None, lyrics=None):
        self.music4all_id = music4all_id
        self.songmeanings_id = songmeanings_id
        self.lyrics = lyrics

    def __repr__(self):
        return f"LyricsCommentData(music4all_id={self.music4all_id}, songmeanings_id={self.songmeanings_id}, lyrics={self.lyrics[:30]}...)"

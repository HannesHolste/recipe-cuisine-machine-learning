import hashlib

_FILE_BUFFER_SIZE_BYTES = 4096

# Adapted from http://stackoverflow.com/questions/3431825/generating-a-md5-checksum-of-a-file
def md5(fname):
    hash = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(_FILE_BUFFER_SIZE_BYTES), b""):
            hash.update(chunk)
    return hash.hexdigest()
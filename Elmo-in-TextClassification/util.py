import errno
import os
import arabic_reshaper
from bidi.algorithm import get_display


def folderpath_handler(file):
    if not os.path.exists(os.path.dirname(file)):
        try:
            os.makedirs(os.path.dirname(file))
        except OSError as exc:
            if exc.errno !=errno.EEXIST:
                raise
def reshape_persiantext_for_display(text):
    reshaped_text = arabic_reshaper.reshape(text)
    try:
        artext = get_display(reshaped_text)
    except AssertionError:
        artext=get_display('error')
    return artext

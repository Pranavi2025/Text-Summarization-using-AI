# Temporary patch for removed cgi module
def parse_header(line):
    return line, {}

def parse_multipart(fp, pdict):
    return {}, {}

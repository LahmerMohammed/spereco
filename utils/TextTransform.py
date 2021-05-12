class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        char_map_str = """
        <SPACE> 1
        ا 2
        ب 3
        ت 4
        ث 5
        ج 6
        ح 7
        خ 8
        د 9
        i 10
        ر 11
        ز 12
        س 13
        ش 14
        ص 15
        ض 16
        ط 17
        ظ 18
        ع 19
        غ 20
        ف 21
        ق 22
        ك 23
        ل 24
        م 25
        ن 26
        ه 27
        و 28
        ي 29
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '


    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                int_ch = self.char_map['<SPACE>']
            else:
                int_ch = self.char_map[c]
            int_sequence.append(int_ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')




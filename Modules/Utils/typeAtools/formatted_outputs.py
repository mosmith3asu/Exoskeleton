
class print_with_header():
    def __init__(self):
        self.header_level = 0
        self.header = ''
        self.logger = []
        self.logoutput = True
        self.enable = True
        self.section_hashes = '##'
        self.section_returns = '\n\n'
    def get_log(self):
        return self.logger
    def process_text(self,texts):
        text = ''
        if type(texts) == list:
            for t in texts: text = text + str(t)
        else: text = texts
        if self.header_level==0: text = text + '...'
        return text

    def process_header(self,level):
        header = ''
        pre_header = ''

        # If header level decresed, add a space at appropriate header level
        if level < self.header_level:
            for i in range(level): header = header + '|\t'
            header = pre_header + '\n'

        # Add header level for this text
        for i in range(level): header = header + '|\t '

        # Assign current header level
        self.header_level = level
        return header

    def process_section(self,args):
        del args[0]

        count =2*len(self.section_hashes)+2
        for arg in args: count += len(arg)
            #header = header.join(['#' for _ in range(len(arg) + 2 * len(self.section_hashes))])
        hash_lead = ''.join(['#' for i in range(count)])
        disp =  self.section_returns + hash_lead + '\n'
        disp = disp + self.section_hashes + ' ' + disp.join(args) + ' ' + self.section_hashes +'\n'+ hash_lead
        return disp

    def display(self,*args):
        if self.enable:
            header = ''
            args = list(args[0])
            if type(args[0])==int and args[0]<10:
                if args[0]==-1:
                    print_output = self.process_section(args) # Section level
                    return print(print_output)

                else:
                    header = self.process_header(args[0])
                    del args[0]

            disp = self.process_text(args)
            print_output = header + disp
            if self.logoutput: self.logger.append(disp)
            print(print_output)



printer = print_with_header()
def printh(*args,ENABLE=None,LOG_OUTPUT=None):
    if LOG_OUTPUT!=None: printer.logoutput = bool(LOG_OUTPUT)
    if ENABLE != None: printer.logoutput = bool(ENABLE)
    printer.display(args)


if __name__ == '__main__':
    printh(0,'Header 1','dfa')
    printh(1, 'Header 2')
    printh(2, 'Header 3')
    printh(2, 'Header 3')

    printh(1, 'Header 2')
    printh(2, 'Header 3')
    printh(2, 'Header 3')

    printh(0, 'Header 1')
    printh('Header 1', 'combo')
    printh(2, 'Header 3','combo')

    printh(-1, 'Section test adgfadg asd')
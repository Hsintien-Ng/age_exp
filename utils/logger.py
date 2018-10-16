import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header


class Logger:
    def __init__(self, name='logger'):
        self.name = name
        self.files = {}
        self.server = ''
        self.sender = ''
        self.sender_password = ''
        self.receiver = ''
        self.email_setup = False

    def open_file(self, log_dir, alias, file_name, file_msg=''):
        """
        Register a log file to logger, which can be accessed or
        manipulated using a handle of alias.
        Note that if there have been already a homonymic alias in
        logger, the latter one will be modified by adding '_new'
        behind the original alias.
        :param log_dir: the file will be created in this directory
        :param alias: a handle to manipulate the open file.Note that
                    this may not be the final alias when homonymic
                    alias occurs.
        :param file_name: the final path of file is log_dir/file_name
        :param file_msg: message that will be printed at the second
                    line of the file, while the first line is alias.
        :return: the alias, probably be modified for the logger have
                a homonymic alias
        """
        while alias in self.files:
            alias = alias + '_new'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file = os.path.join(log_dir, file_name)
        self.files[alias] = open(file, 'w+')
        self.files[alias].write('%s\n%s\n' % (alias, file_msg))
        return alias

    def close_file(self, alias):
        """
        Nothing to be note
        :param alias:
        :return:
        """
        if alias in self.files:
            self.files[alias].close()

    def log_to(self, string, alias=None, log2screen=True):
        """

        :param alias: if alias is None, then log to screen only
        :param string: string to log
        :param log2screen: if False, it will not be print to screen
        :return:
        """
        if alias is not None and alias in self.files:
            self.files[alias].write(string)
            self.files[alias].write('\n')
        if log2screen:
            print(string)

    def flush(self, alias):
        """
        log to file immediately
        :param alias:
        :return:
        """
        self.files[alias].flush()

    def email_setting(self, server, port, sender, password, receiver):
        """
        setting necessary configuration to sent a message
        :param server: SMTP server
        :param port: SMTP port
        :param sender: address to send email
        :param password: password, or authentic code to use
                        SMTP protocol
        :param receiver: using the same receiver as the sender
                    is extremely suggested, for the sent may be
                    treated as spam email and will not be sent.
        :return:
        """
        self.server = server
        self.port = port
        self.sender = sender
        self.sender_password = password
        self.receiver = receiver
        self.email_setup = True

    def send_message_to_me(self, msg):
        """
        send msg as email as the method email_setting has
        defined. The subject of email is self.name
        :param msg: massage you want to send
        :return:
        """
        if not self.email_setup:
            return
        smtp = smtplib.SMTP()
        message = MIMEText(msg, 'plain', 'utf-8')
        message['From'] = Header(self.sender, 'utf-8')
        message['To'] = Header(self.receiver, 'utf-8')

        message['Subject'] = Header(self.name, 'utf-8')
        try:
            smtp.connect(self.server)
            smtp.login(self.sender, self.sender_password)
            smtp.sendmail(self.sender, self.receiver, message.as_string(), )
        except:
            print('Failed to send message!')
        finally:
            smtp.quit()

if __name__ == '__main__':
    work_dir = os.path.join('/', 'home', 'smartcar', 'fuyj', 'Expression')
    log_dir = os.path.join(work_dir, 'log', 'test')
    logger = Logger('test_logger')
    logger.email_setting(server='smtp.qq.com',
                         port=25,
                         sender='yjfu0707@qq.com',
                         password='uhwbcmakldeoddad',
                         receiver='yjfu0707@qq.com')
    alias = logger.open_file(log_dir=log_dir, alias='test',
                             file_name='test.txt',
                             file_msg='just for test')
    alias = logger.open_file(log_dir=log_dir, alias='test',
                             file_name='test2.txt',
                             file_msg='just for test')
    logger.send_message_to_me('yjfu, do you eat')
    logger.log_to('123', alias)
    logger.log_to('456', alias, log2screen=False)


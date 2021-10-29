import pytest
import sys
import io
import os
import time
import threading
import pickle
import pathlib
import math
import json
import random
import pandas as pd

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import listener_analyzer as la

MOCK_FOLLOWERS_FILENAME = 'mock_followers.json'
MOCK_USER_INFO_FILENAME = 'mock_user_info.json'

MOCK_PDS = {}
MOCK_FOLLOWERS = {}
MOCK_USER_INFO = {}

MOCK_IMG = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
MOCK_BANNER_IO = io.BytesIO()
Image.new('RGBA', (400, 400), (0, 100, 0, 127)).save(MOCK_BANNER_IO, 'png')
MOCK_PROF_IO = io.BytesIO()
Image.new('RGBA', (1500, 500), (0, 0, 100, 127)).save(MOCK_PROF_IO, 'png')

PICKLE_IO = io.BytesIO()

THREAD_NUM = os.cpu_count()

@pytest.fixture()
def mock_args(mocker, pytestconfig):
	args = mocker.Mock()
	args.settings = la.DEFAULT_SETTINGS_PATH
	args.actual = pytestconfig.getoption('actual')
	args.update = pytestconfig.getoption('update')
	args.collect = pytestconfig.getoption('collect')
	args.analyze = pytestconfig.getoption('analyze')
	args.post = pytestconfig.getoption('post')
	args.upload = pytestconfig.getoption('upload')
	args.public = pytestconfig.getoption('public')
	args.backup = pytestconfig.getoption('backup')
	args.immediate = pytestconfig.getoption('immediate')
	args.mockf = pytestconfig.getoption('mockf')
	args.verbose = pytestconfig.getoption('vb')
	args.resume = pytestconfig.getoption('resume')
	args.fdir = pytestconfig.getoption('fdir')
	args.save_df = pytestconfig.getoption('save_df')
	args.reuse_result = pytestconfig.getoption('reuse_result')
	return args

def set_mock_followers(members, num, n_range):
	global MOCK_FOLLOWERS
	
	for member in members:
		print('start set mock followers:' + member)
		MOCK_FOLLOWERS[member] = [index for index in random.sample(list(range(n_range)), num)]
		print('end set mock followers:' + member)

def set_mock_members(members):
	global MOCK_PDS
	
	for member in members:
		print('start set mock members:' + member)
		MOCK_PDS[member] = pd.DataFrame(columns=[la.Cols.FOLLOWERED, la.Cols.COLLECT_DATE])
		try:
			MOCK_PDS[member] = MOCK_PDS[member].append(
				pd.DataFrame([[1, time.time()] for f in MOCK_FOLLOWERS[member]], index=MOCK_FOLLOWERS[member],
					columns=[la.Cols.FOLLOWERED, la.Cols.COLLECT_DATE]))
		except:
			print('cannot append member:' + member)
		else:
			print('end set mock members:' + member)

def read_followers(fdir, members):
	global MOCK_FOLLOWERS
	
	for member in members:
		f_path = pathlib.Path(fdir) / (member + '.csv')
		if f_path.is_file():
			print('read followers start:' + member)
			MOCK_FOLLOWERS.update({member:list(pd.read_csv(f_path, sep=',', index_col=0)	\
							.astype({la.Cols.FOLLOWERED:'int8', la.Cols.COLLECT_DATE:'int8'})	\
							.filter(items=[la.Cols.FOLLOWERED], axis=1).index)})
			print('read followers end:' + member)

def thread_start(ths):
	[th.start() for th in ths]
	is_alive = True
	while is_alive:
		is_alive = False
		for th in ths:
			if th.is_alive():
				is_alive = True
				time.sleep(1)
				break

@pytest.mark.freeze_time('2021-04-23 11:22:33')
def test_flow(mocker, mock_args, pytestconfig, monkeypatch, capfd):
	global MOCK_FOLLOWERS
	global MOCK_USER_INFO
	
	expected_result = None
	with open(pytestconfig.getoption('flow'), encoding='utf-8') as f:
		expected_result = f.read()
	
	MOCK_FOLLOWERS = json.load(open(MOCK_FOLLOWERS_FILENAME, 'r', encoding=la.FILE_ENCODING))
	MOCK_USER_INFO = json.load(open(MOCK_USER_INFO_FILENAME, 'r', encoding=la.FILE_ENCODING))
	members = [member['screen_name'] for member in MOCK_FOLLOWERS.values()]
	divs = [(len(members) + i) // THREAD_NUM for i in reversed(range(THREAD_NUM))]
	
	fdir = mock_args.fdir
	if fdir is not None and os.path.isdir(fdir):
		start_index = 0
		ths = []
		for div in divs:
			ths.append(threading.Thread(target=read_followers,	\
				args=(fdir, members[start_index:start_index + div],), daemon=True))
			start_index += div
		thread_start(ths)
	
	mockf = mock_args.mockf
	try:
		if mockf is not None:
			if 1 > int(mockf[0]) or 1 > float(mockf[1]):
				raise Exception('mockf is only allowed positive num')
			elif mock_args.collect or not mock_args.actual:
				start_index = 0
				ths = []
				for div in divs:
					ths.append(threading.Thread(target=set_mock_followers,	\
						args=(members[start_index:start_index + div], int(mockf[0]), int(math.ceil(int(mockf[0]) * float(mockf[1]))),), daemon=True))
					start_index += div
				thread_start(ths)
		
		if not mock_args.actual:
			start_index = 0
			ths = []
			for div in divs:
				ths.append(threading.Thread(target=set_mock_members,	\
					args=(members[start_index:start_index + div],), daemon=True))
				start_index += div
			thread_start(ths)
	except KeyboardInterrupt:
		thread_event.set()
		raise
	
	mocker.patch('argparse.ArgumentParser.parse_args', return_value=mock_args)
	mocker.patch('tweepy.API', side_effect=MockTweepyAPI)
	mocker.patch('tweepy.Cursor', side_effect=MockTweepyCursor)
	[mocker.patch(func, return_value=MockResource()) for func in [
		'googleapiclient.discovery._retrieve_discovery_doc', 'googleapiclient.discovery.build_from_document']]
	mocker.patch('oauth2client.client.Storage.get', return_value=MockStorage())
	mocker.patch('google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file', return_value=MockCredentials())
	mocker.patch('requests.Session.request', side_effect=MockRequest().mock_session_request)
	
	if not mock_args.actual:
		with open(la.DEFAULT_SETTINGS_PATH, 'r', encoding=la.FILE_ENCODING) as f:
			MockOpen.data.update({la.DEFAULT_SETTINGS_PATH:f.read()})
		pickle.dump(MockCredentials(), PICKLE_IO)
		mocker.patch('builtins.open', MockOpen)
		monkeypatch.setattr('subprocess.Popen', MockProcess)
		mocker.patch('pathlib.Path.is_file', return_value=True)
		mocker.patch('pathlib.Path.is_dir', return_value=True)
		mocker.patch('pathlib.Path.mkdir', return_value=None)
		mocker.patch('pathlib.Path.iterdir', return_value=[])
		mocker.patch('os.remove', return_value=None)
		mocker.patch('os.makedirs', return_value=None)
		mocker.patch('shutil.copy2', return_value=None)
		mocker.patch('shutil.rmtree', return_value=None)
		mocker.patch('pandas.read_csv', side_effect=mock_pandas_read_csv)
		mocker.patch('matplotlib.figure.Figure.savefig', return_value=None)
		[mocker.patch(func, return_value=MockImage()) for func in ['PIL.Image.open', 'PIL.Image.new',
			'PIL.Image.init', 'PIL.Image.fromarray', 'PIL.Image.alpha_composite', 'PIL.ImageChops.difference']]
		monkeypatch.setattr('PIL.Image.Image', MockImage)
		monkeypatch.setattr('PIL.ImageDraw.ImageDraw', MockDraw)
		mocker.patch('imageio.core.format.FormatManager.search_read_format', return_value=MockHeader())
		mocker.patch('imageio.core.request.Request._parse_uri', return_value=None)
		monkeypatch.setattr('moviepy.video.VideoClip.ColorClip', MockClip)
		mocker.patch('moviepy.video.VideoClip.VideoClip.on_color', return_value=MockClip())
		mocker.patch('moviepy.video.fx.all.scroll', return_value=MockClip())
		mocker.patch('moviepy.video.VideoClip.VideoClip.write_videofile', return_value=None)
	
	la.main()
	
	out, err = capfd.readouterr()
	print(out)
	
	assert expected_result == out


class MockTweepyAPI:
	followers_ids = ''
	current_id = 0
	current_media_id = 0
	def __init__(self, auth, wait_on_rate_limit=False, proxy=None):
		self.auth = auth
		self.wait_on_rate_limit = wait_on_rate_limit
		self.ids = []
	
	def update_status(self, text=None, status=None, media_ids=None, in_reply_to_status_id=None):
		return self.mock_id_update()
	
	def media_upload(self, filename=None, file=None):
		return self.mock_media_update()
	
	def mock_id_update(self):
		current_id = MockTweepyAPI.current_id
		MockTweepyAPI.current_id += 1
		return MockStatus(id=current_id)
	
	def mock_media_update(self):
		current_media_id = MockTweepyAPI.current_media_id
		MockTweepyAPI.current_media_id += 1
		return MockStatus(media_id=current_media_id)
	
	def get_user(self, id):
		return MockUserInfo(id)

class MockStatus:
	def __init__(self, id=None, media_id=None):
		self.id = id
		self.media_id = media_id

class MockUserInfo:
	def __init__(self, id):
		self._json = MOCK_USER_INFO[str(id)] if str(id) in MOCK_USER_INFO else None

class MockTweepyCursor:
	def __init__(self, followers_ids, id=None, cursor=0):
		self.id = id
		self.cursor = cursor
	
	def pages(self):
		return MockIterator(self.id, self.cursor)

class MockIterator:
	MAX_NUM = 5000
	def __init__(self, id, current_cursor):
		self.id = str(id)
		page_num = int(math.ceil(len(MOCK_FOLLOWERS[self.id]['followers']) / MockIterator.MAX_NUM))
		self.start_index = 0 if -1 == current_cursor	\
			else current_cursor * MockIterator.MAX_NUM if 0 < current_cursor and current_cursor < page_num	\
			else None
		
		self.next_cursor = 0
		if len(MOCK_FOLLOWERS[self.id]['followers']) > MockIterator.MAX_NUM:
			self.next_cursor = 1 if -1 == current_cursor	\
			else current_cursor + 1 if 0 < current_cursor and current_cursor + 1 < page_num	\
			else self.next_cursor
	
	def next(self):
		return MOCK_FOLLOWERS[self.id]['followers'][self.start_index:self.start_index + MockIterator.MAX_NUM]	\
			if self.start_index is not None else []

class MockResource:
	def __init__(self):
		pass
	
	def videos(self):
		return MockVideos()
	
	def users(self):
		return self
	
	def messages(self):
		return self
	
	def send(self, userId, body):
		return self
	
	def execute(self):
		return {}
	
	def threads(self):
		return self
	
	def list(self, userId, q, pageToken):
		return self
	
	def get(self, user_id, id):
		return self

class MockVideos:
	def insert(self, part=None, body=None, media_body=None):
		return MockInsertRequest()

class MockInsertRequest:
	def next_chunk(self):
		return 200, dict(id='mock_id')

class MockStorage:
	def __init__(self, *args, **kwargs):
		self.invalid = False
	
	def authorize(self, *args):
		return None

class MockCredentials:
	def __init__(self, *args):
		self.valid = True
		self.expired = False
		self.refresh_token = False
	
	def run_local_server(self):
		pass

class MockRequest:
	def __init__(self):
		self.count = -1
	
	def mock_session_request(self, method, url=None, headers=None, timeout=None, **kwargs):
		self.count += 1
		return MockRequestResult(content=MOCK_BANNER_IO.getvalue()) if url.startswith('https://pbs.twimg.com/profile_images/')	\
			else MockRequestResult(content=MOCK_PROF_IO.getvalue()) if url.startswith('https://pbs.twimg.com/profile_banners/')	\
			else MockRequestResult(content='{"access_token":""}') if url.startswith('https://oauth2.googleapis.com/token')	\
			else MockRequestResult(text=str(self.count))

class MockRequestResult:
	def __init__(self, content=b'', text=''):
		self.content = content
		self.text = text
		self.status_code = 200

class MockOpen:
	data = {}
	
	def __init__(self, path, mode=None, encoding=None, errors=None, newline=None):
		self.path = str(path)
		self.read = self.mock_read
		self.readline = True
		MockOpen.data.update({self.path:PICKLE_IO.getvalue()}) if self.path.endswith('.pickle') else None
	
	def __enter__(self):
		return self
	
	def __exit__(self, *args):
		return self
	
	def mock_read(self, *args):
		return MockOpen.data[self.path] if self.path in MockOpen.data else ''
	
	def seek(self, *args):
		pass
	
	def tell(self):
		return 1
	
	def write(self, *args):
		pass
	
	def flush(self):
		pass
	
	def close(self):
		pass

class MockProcess:
	def __init__(self, cmd, stdout=None, stderr=None, **kwargs):
		self.cmd = cmd
		MockOpen.data.update(dict(Process=b''))
		self.stdout = MockOpen('Process')
		self.stderr = MockOpen('Process')
	
	def communicate(self):
		if la.DOCKER_CMD == self.cmd[0]:
			return b'mock_id', b''
		if la.Cloud.EXEC_CMD == self.cmd[0]:
			return b'mock_upload', b''
		if 'ffmpeg' in self.cmd[0]:
			return b'ffmpeg', b'Duration: 00:00:01.000'
		return b'', b''
	
	def terminate(self):
		pass
	
	def wait(self):
		pass

class MockImage:
	core = None
	
	def __init__(self, *args, **kwargs):
		self.size = (1, 1)
		self.format = None
		self.width = 1
		self.height = 1
		self.shape = [1, 1]
	
	def resize(self, *args):
		return self
	
	def copy(self, *args):
		return self
	
	def paste(self, *args):
		return self
	
	def crop(self, *args):
		return self
	
	def getbbox(self):
		return self
	
	def convert(self, *args):
		return self
	
	def rotate(self, *args):
		return self
	
	def save(self, *args, **kwargs):
		pass
	
	def close(self):
		pass

class MockDraw:
	def __init__(self, *args, **kwargs):
		pass
	
	def text(self, *args, **kwargs):
		pass

class MockHeader:
	def get_reader(self, *args):
		return self
	
	def __enter__(self):
		return self
	
	def __exit__(self, *args):
		return self
	
	def get_data(self, *args):
		return MockImage()

class MockClip:
	def __init__(self, *args, **kwargs):
		self.size = (1, 1)
		self.start = 1
		self.end = 1
		self.duration = 1
		self.audio = None
		self.mask = self
		self.pos = None
		self.ismask = True
	
	def margin(self, *args, **kwargs):
		return self
	
	def resize(self, *args, **kwargs):
		return self
	
	def set_position(self, *args, **kwargs):
		return self
	
	def set_duration(self, *args, **kwargs):
		return self
	
	def set_start(self, *args, **kwargs):
		return self
	
	def set_end(self, *args, **kwargs):
		return self
	
	def add_mask(self, *args, **kwargs):
		return self
	
	def fadein(self, *args, **kwargs):
		return self
	
	def fadeout(self, *args, **kwargs):
		return self
	
	def loop(self):
		return self

def mock_pandas_read_csv(path, sep=None, index_col=None):
	basename = os.path.basename(path)
	
	return MOCK_PDS[os.path.splitext(basename)[0]]	\
			if os.path.splitext(basename)[0] in MOCK_PDS.keys()	\
		else pd.DataFrame(columns=[la.Cols.FOLLOWERED, la.Cols.COLLECT_DATE])

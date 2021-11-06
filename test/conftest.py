from distutils.util import strtobool

def pytest_addoption(parser):
	parser.addoption('--flow', required=True)
	parser.addoption('--actual', action='store_true')
	parser.addoption('--update', action='store_true')
	parser.addoption('--collect', type=strtobool, default=False)
	parser.addoption('--analyze', type=strtobool, default=True)
	parser.addoption('--post', type=strtobool, default=False)
	parser.addoption('--upload', action='store_true')
	parser.addoption('--public', action='store_true')
	parser.addoption('--backup', type=strtobool, default=False)
	parser.addoption('--immediate', action='store_true')
	parser.addoption('--mockf', nargs=2)
	parser.addoption('--vb', type=strtobool, default=True)
	parser.addoption('--fdir')
	parser.addoption('--save_df', action='store_true')
	parser.addoption('--reuse_result', action='store_true')

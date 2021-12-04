from distutils.util import strtobool

def pytest_addoption(parser):
	parser.addoption('--f', required=True)
	parser.addoption('--act', action='store_true')
	parser.addoption('--u', action='store_true')
	parser.addoption('--c', type=strtobool, default=False)
	parser.addoption('--a', type=strtobool, default=True)
	parser.addoption('--p', type=strtobool, default=False)
	parser.addoption('--ul', action='store_true')
	parser.addoption('--pub', action='store_true')
	parser.addoption('--b', type=strtobool, default=False)
	parser.addoption('--i', action='store_true')
	parser.addoption('--mf', nargs=2)
	parser.addoption('--vb', type=strtobool, default=True)
	parser.addoption('--fd')
	parser.addoption('--sd', action='store_true')
	parser.addoption('--rr', action='store_true')
	parser.addoption('--us', action='store_true')

import os
import zipfile
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
#source https://botproxy.net/docs/how-to/setting-chromedriver-proxy-auth-with-selenium-using-python/

def build_proxy_extension(host,port,user,password,pluginfile='proxy_auth_plugin.zip'):

    manifest_json = """
    {
	"version": "1.0.0",
	"manifest_version": 2,
	"name": "Chrome Proxy",
	"permissions": [
	    "proxy",
	    "tabs",
	    "unlimitedStorage",
	    "storage",
	    "<all_urls>",
	    "webRequest",
	    "webRequestBlocking"
	],
	"background": {
	    "scripts": ["background.js"]
	},
	"minimum_chrome_version":"22.0.0"
    }
    """

    background_js = """
    var config = {
	    mode: "fixed_servers",
	    rules: {
	      singleProxy: {
		scheme: "http",
		host: "%s",
		port: parseInt(%s)
	      },
	      bypassList: ["localhost"]
	    }
	  };

    chrome.proxy.settings.set({value: config, scope: "regular"}, function() {});

    function callbackFn(details) {
	return {
	    authCredentials: {
		username: "%s",
		password: "%s"
	    }
	};
    }

    chrome.webRequest.onAuthRequired.addListener(
		callbackFn,
		{urls: ["<all_urls>"]},
		['blocking']
    );
    """ % (host,port,user,password)

    with zipfile.ZipFile(pluginfile, 'w') as zp:
        zp.writestr("manifest.json", manifest_json)
        zp.writestr("background.js", background_js)

    return pluginfile


def get_chromedriver(proxy_file=None, headless=True,path=None):
    if not path:
        path = os.path.dirname(os.path.abspath(__file__))
    #path = os.path.dirname(os.getcwd())    
    chrome_options = webdriver.ChromeOptions()

    if proxy_file:
        chrome_options.add_extension(proxy_file)

    if headless:
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')    
    #driver = webdriver.Chrome(
    #    os.path.join(path, 'chromedriver'),
    #    chrome_options=chrome_options)

    service = Service(executable_path=os.path.join(path, 'chromedriver'))
    driver = webdriver.Chrome(
        service=service,
        options=chrome_options)

    return driver

def main():
    proxy = build_proxy_extension(host='somehost'
				  ,port=1234
				  ,user='username'
				  ,password='password'
				 )
    driver = get_chromedriver(proxy_file=proxy)
    #driver.get('https://www.google.com/search?q=my+ip+address')
    driver.get('https://httpbin.org/ip')

if __name__ == '__main__':
    main()
import urllib.request
import sys

points = 1
if len(sys.argv) > 1:
    points = int(sys.argv[1])

aritcleUrl = 'http://blog.csdn.net/shine19930820/article/digg?ArticleId=71713680'
point_header = {
    'Accept' : '*/*',
    'Cookie' : 'your cookie'
    'Host':'blog.csdn.net',
    'Referer' : 'http://blog.csdn.net/shine19930820/article/details/71713680',
    'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.71 Safari/537.36'
}

for i in range(points):
    point_request = urllib.request.Request(aritcleUrl, headers = point_header)
    point_response = urllib.request.urlopen(point_request)

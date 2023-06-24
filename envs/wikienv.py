import ast
import json
import time
import gym
import re
import requests
import spacy
from bs4 import BeautifulSoup
from datetime import datetime

# import wikipedia

def clean_str(p):
	return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


class textSpace(gym.spaces.Space):
  def contains(self, x) -> bool:
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)
  
def get_oldid_before(page_title):
    url = 'https://en.wikipedia.org/w/api.php'

    date = datetime.strptime('2017-10-01', '%Y-%m-%d')

    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'revisions',
        'titles': page_title,
        'rvlimit': 1,
        'rvdir': 'older',
        'rvstart': date.strftime('%Y%m%d%H%M%S')
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        pages = data['query']['pages']
        for page in pages.values():
            if 'revisions' in page:
                oldid = page['revisions'][0]['revid']
                return oldid
    return None


def search_wiki(term):
	url = 'https://en.wikipedia.org/w/api.php'

	params = {
		'action': 'query',
		'format': 'json',
		'list': 'search',
		'srsearch': term,
	}

	response = requests.get(url, params=params)
	entities = []
	for item in response.json()['query']['search']:
		entities.append(item['title'])
	return entities


class WikiEnv(gym.Env):

	sent_tokenizer = spacy.load("en_core_web_sm")

	def __init__(self):
		"""
		Initialize the environment.
		"""
		super().__init__()
		self.page = None  # current Wikipedia page
		self.obs = None  # current observation
		self.lookup_keyword = None  # current lookup keyword
		self.lookup_list = None  # list of paragraphs containing current lookup keyword
		self.lookup_cnt = None  # current lookup index
		self.steps = 0  # current number of steps
		self.answer = None  # current answer from the agent
		self.observation_space = self.action_space = textSpace()
		self.search_time = 0
		self.num_searches = 0
    
	def _get_obs(self):
		return self.obs

	def _get_info(self):
		return {"steps": self.steps, "answer": self.answer}

	def reset(self, seed=None, return_info=False, options=None):
		# We need the following line to seed self.np_random
		# super().reset(seed=seed)
		self.obs = ("Interact with Wikipedia using search[], lookup[], and "
					"finish[].\n")
		self.page = None
		self.lookup_keyword = None
		self.lookup_list = None
		self.lookup_cnt = None
		self.steps = 0
		self.answer = None
		observation = self._get_obs()
		info = self._get_info()
		return (observation, info) if return_info else observation

	def construct_lookup_list(self, keyword):
		# find all paragraphs
		if self.page is None:
			return []
		paragraphs = self.page.split("\n")
		paragraphs = [p.strip() for p in paragraphs if p.strip()]

		# find all sentence
		sentences = []
		for p in paragraphs:
			sentences += p.split('. ')
		sentences = [s.strip() + '.' for s in sentences if s.strip()]

		parts = sentences
		parts = [p for p in parts if keyword.lower() in p.lower()]
		return parts

	@staticmethod
	def get_page_obs(page):
		# find all paragraphs
		paragraphs = page.split("\n")
		paragraphs = [p.strip() for p in paragraphs if p.strip()]

		# find all sentence
		sentences = []
		for p in paragraphs:
			doc = WikiEnv.sent_tokenizer(p)
			sentences += [sent.text for sent in doc.sents]
		sentences = [s.strip() + '.' for s in sentences if s.strip()]
		return ' '.join(sentences[:5])

    # ps = page.split("\n")
    # ret = ps[0]
    # for i in range(1, len(ps)):
    #   if len((ret + ps[i]).split(" ")) <= 50:
    #     ret += ps[i]
    #   else:
    #     break
    # return ret

	def search_step(self, entity):
		entity = re.sub(r"\s*\(redirect[^)]*\)", "", entity)

		old_time = time.time()
		found_entities = search_wiki(entity)
		self.search_time += time.time() - old_time

		if entity in found_entities:
			search_url = f"https://en.wikipedia.org/w/index.php?search={entity}"
			response_text = requests.get(search_url).text

			soup = BeautifulSoup(response_text, features="html.parser")
			page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
			if any("may refer to:" in p for p in page):
				self.search_step("[" + entity + "]")
			else:

				oldid = get_oldid_before(entity)

				if oldid is None:
					return False
				
				page_url = f"https://en.wikipedia.org/w/index.php?title={entity}&oldid={oldid}"
				old_response_text = requests.get(page_url).text
				old_soup = BeautifulSoup(old_response_text, features="html.parser")
				old_page = [p.get_text().strip() for p in old_soup.find_all("p") + old_soup.find_all("ul")]

				self.page = ""
				for p in old_page[1:]:
					if len(p.split(" ")) > 2:
						self.page += clean_str(p)
						if not p.endswith("\n"):
							self.page += "\n"

				self.obs = self.get_page_obs(self.page)
				self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

				return True
		else:
			self.obs = "Could not find exact match of \"{}\". Available entities in the database: {}. The search API only returns the paragraph if the search query matches every character to an entity in the database.".format(entity, ', '.join('"{}"'.format(e) for e in found_entities[:5]))
		
		self.num_searches += 1

		return False
  
	def step(self, action):
		reward = 0
		done = False
		action = action.strip()
		if self.answer is not None:  # already finished
			done = True
			return self.obs, reward, done, self._get_info()
		
		if action.startswith("search[") and action.endswith("]"):
			entity = action[len("search["):-1]
			# entity_ = entity.replace(" ", "_")
			# search_url = f"https://en.wikipedia.org/wiki/{entity_}"
			self.search_step(entity)
		elif action.startswith("lookup[") and action.endswith("]"):
			keyword = action[len("lookup["):-1]
			if self.lookup_keyword != keyword:  # reset lookup
				self.lookup_keyword = keyword
				self.lookup_list = self.construct_lookup_list(keyword)
				self.lookup_cnt = 0
			if self.lookup_cnt >= len(self.lookup_list):
				self.obs = "No more results.\n"
			else:
				self.obs = f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + self.lookup_list[self.lookup_cnt]
				self.lookup_cnt += 1
		elif action.startswith("finish[") and action.endswith("]"):
			answer = action[len("finish["):-1]
			self.answer = answer
			done = True
			self.obs = f"Episode finished, reward = {reward}\n"
		elif action.startswith("think[") and action.endswith("]"):
			self.obs = "Nice thought."
		else:
			self.obs = "Invalid action: {}".format(action)
		self.steps += 1
		return self.obs, reward, done, self._get_info()
  
	def get_time_info(self):
		speed = self.search_time / self.num_searches if self.num_searches else 0
		return {
			"call_speed": speed,
			"call_time": self.search_time,
			"num_calls": self.num_searches,
		}


if __name__ == '__main__':
	wiki = WikiEnv()
	wiki.search_step("Big Stone Gap, Virginia")
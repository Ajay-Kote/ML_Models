
"""
url_feature_extractor.py
Research-oriented lexical URL feature extractor.
"""

import math
import re
import ipaddress
from urllib.parse import urlparse
import tldextract


class URLFeatureExtractor:

    SUSPICIOUS_KEYWORDS = [
        "login","signin","verify","secure","update","account",
        "password","bank","payment","wallet","crypto",
        "bitcoin","invoice","admin","support","confirm","auth"
    ]

    BRAND_KEYWORDS = [
        "google","paypal","amazon","facebook","apple",
        "microsoft","github","netflix","instagram",
        "linkedin","discord","steampowered"
    ]

    SHORTENERS = {
        "bit.ly","tinyurl.com","goo.gl","t.co",
        "ow.ly","is.gd","buff.ly"
    }

    HIGH_RISK_TLDS = {
        "xyz","top","tk","cf","ml","gq",
        "click","work","support"
    }

    def __init__(self, url:str):
        self.original_url = str(url).strip()
        self.url = self._normalize_url(self.original_url)
        self.parsed = urlparse(self.url)
        self.ext = tldextract.extract(self.url)

    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        Normalize URLs to match training data format.
        Legitimate URLs in the dataset are bare domains without trailing slashes.
        """
        parsed = urlparse(url)
        if parsed.path == "/" and not parsed.query and not parsed.fragment:
            return url.rstrip("/")
        return url

    def _entropy(self):
        if not self.url:
            return 0
        probs=[self.url.count(c)/len(self.url) for c in set(self.url)]
        return -sum(p*math.log2(p) for p in probs)

    def _max_run(self, pattern):
        m = re.findall(pattern,self.url)
        return max((len(x) for x in m), default=0)

    def _is_shortened_url(self):
        host = self.parsed.netloc.lower().split(":")[0]
        if host.startswith("www."):
            host = host[4:]

        return int(
            any(
                host == shortener or host.endswith("." + shortener)
                for shortener in self.SHORTENERS
            )
        )

    @staticmethod
    def _levenshtein(a, b):
        if a == b:
            return 0
        if not a or not b:
            return max(len(a), len(b))

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
            prev = curr
        return prev[-1]

    def _brand_similarity(self):
        domain = self.ext.domain.lower()

        if not domain:
            return 999, 0

        distances = [self._levenshtein(domain, brand) for brand in self.BRAND_KEYWORDS]
        min_distance = min(distances)

        # Close to a brand but not an exact match and not just a brand
        # used as a normal word inside a much longer domain.
        typosquat = int(0 < min_distance <= 2 and len(domain) <= 20)

        return min_distance, typosquat

    def _brand_domain_match(self):
        domain = self.ext.domain.lower()

        exact = int(domain in self.BRAND_KEYWORDS)

        # Brand name appears somewhere in the host, but the registrable
        # domain itself isn't that brand - e.g. "paypal-secure-login" or
        # "accounts-google" as the domain. This is the impersonation
        # pattern; it's distinct from a brand's own domain or a genuine
        # subdomain of it (e.g. accounts.google.com still has domain
        # "google", so exact=1 there too).
        mismatch = int(
            not exact and any(brand in domain for brand in self.BRAND_KEYWORDS)
        )

        return exact, mismatch

    def _count_keyword_matches(self, keywords):
        host = self.parsed.netloc.lower().split(":")[0]
        path = self.parsed.path.lower()
        query = self.parsed.query.lower()

        searchable = " ".join(
            part for part in [host, path, query] if part
        )

        return sum(keyword in searchable for keyword in keywords)

    def extract(self):
        host=self.parsed.netloc
        path=self.parsed.path
        query=self.parsed.query
        frag=self.parsed.fragment

        letters=sum(c.isalpha() for c in self.url)
        digits=sum(c.isdigit() for c in self.url)
        upper=sum(c.isupper() for c in self.url)
        lower=sum(c.islower() for c in self.url)
        special=sum(not c.isalnum() for c in self.url)

        total=max(len(self.url),1)

        feats={
            # length
            "URL_Length":len(self.url),
            "Domain_Length":len(self.ext.domain),
            "Host_Length":len(host),
            "Path_Length":len(path),
            "Query_Length":len(query),
            "Fragment_Length":len(frag),

            # counts
            "Letters":letters,
            "Digits":digits,
            "Uppercase":upper,
            "Lowercase":lower,
            "Special_Characters":special,
            "Dots":self.url.count("."),
            "Hyphens":self.url.count("-"),
            "Underscores":self.url.count("_"),
            "Slashes":self.url.count("/"),
            "QuestionMarks":self.url.count("?"),
            "Equals":self.url.count("="),
            "Ampersands":self.url.count("&"),
            "AtSymbols":self.url.count("@"),
            "Percents":self.url.count("%"),
            "Colons":self.url.count(":"),

            # ratios
            "Digit_Ratio":digits/total,
            "Letter_Ratio":letters/total,
            "Upper_Ratio":upper/total,
            "Lower_Ratio":lower/total,
            "Special_Ratio":special/total,

            # structure
            "HTTPS": int(self.url.lower().startswith("https")),
            "Subdomains":0 if self.ext.subdomain=="" else len(self.ext.subdomain.split(".")),
            "Port_Present":int(self.parsed.port is not None),

            # ip
            "IP_Address":0,

            # entropy
            "Entropy":round(self._entropy(),4),
            "Unique_Characters":len(set(self.url)),
            "Character_Diversity":len(set(self.url))/total,

            # keywords
            "Suspicious_Keyword_Count":0,
            "Brand_Count":0,

            # typosquatting
            "Brand_Edit_Distance":0,
            "Typosquat_Suspected":0,
            "Brand_Exact_Domain":0,
            "Brand_Domain_Mismatch":0,

            # url shortener
            "Shortened_URL":0,

            # tld
            "High_Risk_TLD":int(self.ext.suffix in self.HIGH_RISK_TLDS),

            # token stats
            "Token_Count":0,
            "Average_Token_Length":0,
            "Maximum_Token_Length":0,

            # consecutive
            "Max_Consecutive_Digits":self._max_run(r"\d+"),
            "Max_Consecutive_Letters":self._max_run(r"[A-Za-z]+"),
            "Max_Consecutive_Special":self._max_run(r"[^A-Za-z0-9]+"),
        }

        try:
            ipaddress.ip_address(host.split(":")[0])
            feats["IP_Address"]=1
        except Exception:
            pass

        lower_url=self.url.lower()

        feats["Suspicious_Keyword_Count"]=self._count_keyword_matches(
            self.SUSPICIOUS_KEYWORDS
        )

        feats["Brand_Count"]=self._count_keyword_matches(
            self.BRAND_KEYWORDS
        )

        feats["Shortened_URL"]=self._is_shortened_url()

        brand_distance, typosquat = self._brand_similarity()
        feats["Brand_Edit_Distance"]=brand_distance
        feats["Typosquat_Suspected"]=typosquat

        brand_exact, brand_mismatch = self._brand_domain_match()
        feats["Brand_Exact_Domain"]=brand_exact
        feats["Brand_Domain_Mismatch"]=brand_mismatch

        tokens=[t for t in re.split(r"[./?=&:_-]+",lower_url) if t]
        if tokens:
            feats["Token_Count"]=len(tokens)
            feats["Average_Token_Length"]=sum(map(len,tokens))/len(tokens)
            feats["Maximum_Token_Length"]=max(map(len,tokens))

        return feats


if __name__=="__main__":
    url=input("Enter URL: ")
    f=URLFeatureExtractor(url).extract()
    for k,v in f.items():
        print(f"{k}: {v}")

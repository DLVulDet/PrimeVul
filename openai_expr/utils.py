# **Standard Classification Prompt**

SYS_INST = "You are a security expert that is good at static program analysis."

PROMPT_INST = """Please analyze the following code:
```
{func}
```
Please indicate your analysis result with one of the options: 
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability. 

Only reply with one of the options above. Do not include any further information.
"""


# **Chain of Thought Prompt**

PROMPT_INST_COT = """Please analyze the following code:
```
{func}
```
Please indicate your analysis result with one of the options: 
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability. 

Make sure to include one of the options above "explicitly" (EXPLICITLY!!!) in your response.
Let's think step-by-step.
"""

ONESHOT_USER = """Please analyze the following code:
```
static char *clean_path(char *path)
{
        char *ch;
        char *ch2;
        char *str;
        str = xmalloc(strlen(path) + 1);
        ch = path;
        ch2 = str;
        while (true) {
                *ch2 = *ch;
                ch++;
                ch2++;
                if (!*(ch-1))
                        break;
                while (*(ch - 1) == '/' && *ch == '/')
                        ch++;
        }
        /* get rid of trailing / characters */
        while ((ch = strrchr(str, '/'))) {
                if (ch == str)
                        break;
                if (!*(ch+1))
                        *ch = 0;
                else
                        break;
        }
        return str;
}
```
Please indicate your analysis result with one of the options: 
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability. 

Only reply with one of the options above. Do not include any further information.
"""

ONESHOT_ASSISTANT = "NO"

TWOSHOT_USER = """Please analyze the following code:
```
int64 ClientUsageTracker::GetCachedHostUsage(const std::string& host) {
   HostUsageMap::const_iterator found = cached_usage_.find(host);
   if (found == cached_usage_.end())
     return 0;

  int64 usage = 0;
  const UsageMap& map = found->second;
  for (UsageMap::const_iterator iter = map.begin();
       iter != map.end(); ++iter) {
    usage += iter->second;
  }
  return usage;
}

```
Please indicate your analysis result with one of the options: 
(1) YES: A security vulnerability detected.
(2) NO: No security vulnerability. 

Only reply with one of the options above. Do not include any further information.
"""

TWOSHOT_ASSISTANT = "YES"

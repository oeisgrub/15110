"""
15-110 Hw6 - Tweet Analytics Project
Name: Oskar Eisgruber
AndrewID: oeisgrub
"""

project = "TweetAnalysis" # don't edit this

#### CHECK-IN 1 ####

"""
First, install all the needed libraries. These installations should run without 
an error if you have completed the installation instructions correctly.
"""
import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pandas as pd


"""
Write a function makeDataFrame(filename) which takes in a string filename of a
CSV file and returns a pandas dataframe.
Hint: Use the Data Analysis lecture notes.
"""
def makeDataFrame(filename):
    df = pd.read_csv(filename,delimiter=",", quotechar='"')
    return df

"""
Write a function parseName(from_string) that takes a string of the form:
From: <FirstName> <Lastname> (<Position> from <State>)
and returns the name (first and last).
Hint: what comes before and after the name that you could count or split on?
"""
def parseName(from_string):
    nameString = ""
    labels = from_string.split(" ")
    if len(labels) < 6:
        firstName = labels[1]
        return firstName 
    else:
        firstName = labels[1]
        lastName = labels[2]
        fullName = firstName + " " + lastName
        return fullName

"""
Write a function parsePosition(from_string) that takes a string of the form:
From: <FirstName> <Lastname> (<Position> from <State>)
and returns the position.
Hint: what comes before and after the position that you could find or split on?
"""
def parsePosition(from_string):
    specifics = from_string.split("(")
    labels = specifics[1].split(" ")
    position = labels[0]
    state = labels[2]
    return position





"""
Write a function parseState(from_string) that takes a string of the form:
From: <FirstName> <Lastname> (<Position> from <State>)
and returns the state.
Hint: what comes before and after the state that you could find or split on?
"""
def parseState(from_string):
    x = from_string.index("from")
    y = from_string.index(")")
    state = from_string[x+5:y]
    return state

"""
Write a function getRegionFromState(state_df, state) that takes a state_df dataframe
and a string of a state name to search for and return the corresponding region. 
The state_df has rows where each state is in the column 'State' and its 
corresponding region of the US, like Northeast or South, is in the 'Region' column.

Often, when looking up values in a dataframe, you know the value of one column, 
and you want to look up the associated value of a different column (like knowing 
a person's name and wanting to look up their phone number). To do this, use the code:

df.loc[df['known column name'] == 'known value to match', 'column name to return']. 

Use this code to get the row associated with the state in the state dataframe 
using the correct column name and state value. Then, get the value of this cell 
in the dataframe by using row.values[0], and return this value.
"""
def getRegionFromState(state_df, state):
    row = state_df.loc[state_df['State'] == state, 'Region']
    if len(row.values) == 0:
        return None
    else:
        return row.values[0]
"""
Write a function addColumns(data, state_df) that takes in a dataframe data and a 
dataframe state_df, and destructively adds four new columns to data based on the 
data based on the functions you wrote above. Return None at the end of the function.

1) In order to add columns to a dataframe, you must make lists of all the values
   to add (one for each new column). Create four new empty lists for names, 
   positions, states, and regions of the tweeters.
2) Uses iterrows() to iterate through each index and row in data. For each row, 
   the function should:
    a) get the value in the column 'label' (the code looks the same as a dictionary 
       with key 'label')
    b) call parseName, parsePosition, and parseState on that value.
    c) call getRegionFromState with state_df and the state you parsed to get its region.
    d) append each of the values to their respective list.
3) After the end of the for loop, set data['Name'], data['Position'], data['State'],
and data['Region'] to the respective lists.
Return None
"""
def addColumns(data, state_df):
    names = [] 
    positions = []
    states = []
    regions = []
    for index,row in data.iterrows():
        x = row['label']
        names += [parseName(x)]
        positions += [parsePosition(x)]
        states += [parseState(x)]
        if getRegionFromState(state_df,parseState(x)) == "":
            regions += [""]
        else:
            regions += [getRegionFromState(state_df,parseState(x))]
    data['Name'] = names
    data['Position'] = positions
    data['State'] = states
    data['Region'] = regions
    return None


def do_week1():
    df = makeDataFrame("politicaldata.csv")
    state_df = makeDataFrame("statemappings.csv")
    addColumns(df,state_df)
    print("Updated dataframe:")
    print(df)

#### WEEK 1 TESTS ####

def testMakeDataFrame():
    print("Testing makeDataFrame()...", end="")
    assert(type(makeDataFrame("politicaldata.csv")) == pd.core.frame.DataFrame)
    print("... done!")

def testParseName():
    print("Testing parseName()...", end="")
    assert(parseName("From: Steny Hoyer (Representative from Maryland)") == "Steny Hoyer")
    assert(parseName("From: Mitch (Senator from Kentucky)") == "Mitch")
    print("...done!")

def testParsePosition():
    print("Testing parsePosition()...", end="")
    assert(parsePosition("From: Steny Hoyer (Representative from Maryland)") == "Representative")
    assert(parsePosition("From: Mitch (Senator from Kentucky)") == "Senator")
    print("...done!")

def testParseState():
    print("Testing parseState()...", end="")
    assert(parseState("From: Steny Hoyer (Representative from Maryland)") == "Maryland")
    assert(parseState("From: Mitch (Senator from Kentucky)") == "Kentucky")
    print("...done!")

def testGetRegionFromState():
    print("Testing getRegionFromState()...", end="")
    state_df = makeDataFrame("statemappings.csv")
    assert(str(getRegionFromState(state_df, "California")) == "West")
    assert(str(getRegionFromState(state_df, "Maine")) == "Northeast")
    print("...done!")

def testWeek1():
    testMakeDataFrame()
    testParseName()
    testParsePosition()
    testParseState()
    testGetRegionFromState()

testWeek1()
do_week1()

#### CHECK-IN 2 ####

"""
Edit the function findSentiment (classifier, tweet) to return "positive" if the 
classifier predicts the tweet is positive, "negative" if it is negative, and 
"neutral" if it is neutral. The first line of the  code already runs a classifier 
on the tweet and receives a float score. You should add code to check if the score 
is less than -0.1 and if so return "negative"; if greater than 0.1, return 
"positive"; and otherwise return "neutral".
"""
def findSentiment(classifier, tweet):
    score = classifier.polarity_scores(tweet)['compound']
    if score < -0.1:
        return "negative"
    if score > 0.1:
        return "positive"
    else:
        return "neutral"

"""
Edit the function addSentimentColumn(data) to perform a similar function to 
addColumns, but with sentiment.

Like last time, you should create a new empty list of sentiments. Then, use 
iterrows() to iterate through each index and row of the dataframe data. For each 
row, get the value of the 'text' column which represents the tweet, pass the 
tweet and the classifier to findSentiment, and append the result to the sentiments 
list. After the loop is finished, it should set data['Sentiment'] equal to the 
list of sentiments and return None.
"""
def addSentimentColumn(data):
    sentiments = []
    classifier = SentimentIntensityAnalyzer()
    for index,row in data.iterrows():
        x = row['text']
        sentiments.append(findSentiment(classifier, x))
    data['Sentiment'] = sentiments
    return None

"""
Write a function getNegSentimentByState(data) that creates a dictionary of tweet
sentiment by the state of the tweeter. It should return a dictionary that maps 
each state to the number of messages from that state that had a negative sentiment.

To solve this problem, recall how we can generate dictionaries that map values to 
counts. Do the same thing here, but use row['Sentiment'] and row['State'] to access 
your values.
"""
def getNegSentimentByState(data):
    StateSent = {}
    for index, row in data.iterrows():
        if row['Sentiment'] == "negative":
            if row['State'] not in StateSent:
                StateSent[row['State']] = 0
            StateSent[row['State']] += 1
    return StateSent

"""
Write a function getAttacksByState(data) that creates a dictionary of number of 
tweet attacks by the state of the tweeter. This should use the same process as
getNegSentimentByState(), except that you'll check whether the column 'message' 
is set to 'attack' instead of checking if 'Sentiment' is 'negative'.
"""
def getAttacksByState(data):
    StateAttack = {}
    for index, row in data.iterrows():
        if row['message'] == "attack":
            if row['State'] not in StateAttack:
                StateAttack[row['State']] = 0
            StateAttack[row['State']] += 1
    return StateAttack

"""
Write a function getPartisanByState(data) that creates a dictionary of number of
partisan tweets by the state of the tweeter. This should use the same process as
getNegSentimentByState(), except that you'll check whether the column 'bias' is 
set to 'partisan' instead of checking if 'Sentiment' is 'negative'.
"""
def getPartisanByState(data):
    StatePart = {}
    for index, row in data.iterrows():
        if row['bias'] == "partisan":
            if row['State'] not in StatePart:
                StatePart[row['State']] = 0
            StatePart[row['State']] += 1
    return StatePart

"""
Write a function getMessagesByRegion(data) that returns a nested dictionary. The
keys of the outer dictionary should be regions, which each map to an inner
dictionary. The keys of the inner dictionary should be message types (like 
'attack' from before). Each message type should map to the number of times it
occurred in that region in the dataset.
"""
def getMessagesByRegion(data):
    regions = {}
    for index, row in data.iterrows():
        if row['Region'] not in regions:
            regions[row["Region"]] = {}
        if row['message'] not in regions[row['Region']]:
            regions[row['Region']][row['message']] = 0
        regions[row['Region']][row['message']] += 1
    print (regions[row['Region']])
    return regions

"""
Finally, write a function getAudienceByRegion(data) that returns a nested
dictionary. The keys of the outer dictionary should again be regions, which each
map to an inner dictionary. The keys of the inner dictionary should be audience
types, from the column 'audience'. Each message type should map to the number of
times it occurs in that region.
"""
def getAudienceByRegion(data):
    regions = {}
    for index, row in data.iterrows():
        if row['Region'] not in regions:
            regions[row["Region"]] = {}
        if row['audience'] not in regions[row['Region']]:
            regions[row['Region']][row['audience']] = 0
        regions[row['Region']][row['audience']] += 1
    print (regions[row['Region']])
    return regions


def do_week2():
    df = makeDataFrame("politicaldata.csv")
    state_df = makeDataFrame("statemappings.csv")
    addColumns(df, state_df)
    addSentimentColumn(df)

    negSentiments = getNegSentimentByState(df)
    attacks = getAttacksByState(df)
    parisanship = getPartisanByState(df)
    messages = getMessagesByRegion(df)
    audiences = getAudienceByRegion(df)

#### WEEK 2 TESTS ####

def testFindSentiment():
    print("Testing findSentiment()...", end="")
    classifier = SentimentIntensityAnalyzer()
    assert(findSentiment(classifier, "great") == "positive")
    assert(findSentiment(classifier, "bad") == "negative")
    assert(findSentiment(classifier, "") == "neutral")
    print("...done!")

def testAddSentimentColumn():
    print("TO TEST SENTIMENT COLUMN, CHECK THE OUTPUT BELOW")
    df = makeDataFrame("politicaldata.csv")
    addSentimentColumn(df)
    print(df["Sentiment"])
    print("IS THERE A SENTIMENT IN EVERY ROW?")

def testGetNegSentimentByState(df):
    print("Testing getNegSentimentByState()...", end="")
    d = getNegSentimentByState(df)
    assert(len(d) == 49)
    assert(d["Pennsylvania"] == 48)
    assert(d["North Dakota"] == 3)
    assert(d["Louisiana"] == 20)
    print("...done!")

def testGetAttacksByState(df):
    print("Testing getAttacksByState()...", end="")
    d = getAttacksByState(df)
    assert(len(d) == 37)
    assert(d["Pennsylvania"] == 9)
    assert(d["Maryland"] == 4)
    assert(d["Nevada"] == 1)
    print("...done!")

def testGetPartisanByState(df):
    print("Testing getPartisanByState()...", end="")
    d = getPartisanByState(df)
    assert(len(d) == 50)
    assert(d["Pennsylvania"] == 40)
    assert(d["Maryland"] == 44)
    assert(d["Nevada"] == 10)
    print("...done!")

def testGetMessagesByRegion(df):
    print("Testing getMessagesByRegion()...", end="")
    d = getMessagesByRegion(df)
    assert(len(d) == 4)
    assert(len(d["South"]) == 9)
    assert(d["South"]["policy"] == 563)
    assert(d["Northeast"]["attack"] == 23)
    print("...done!")

def testGetAudienceByRegion(df):
    print("Testing getAudienceByRegion()...", end="")
    d = getAudienceByRegion(df)
    assert(len(d) == 4)
    assert(len(d["South"]) == 2)
    assert(d["South"]["national"] == 1561)
    assert(d["Midwest"]["constituency"] == 265)
    assert(d["Northeast"]["national"] == 682)
    print("...done!")

def testWeek2():
    testFindSentiment()
    testAddSentimentColumn()
    
    df = makeDataFrame("politicaldata.csv")
    state_df = makeDataFrame("statemappings.csv")
    addColumns(df, state_df)
    addSentimentColumn(df)
    
    testGetNegSentimentByState(df)
    testGetAttacksByState(df)
    testGetPartisanByState(df)
    testGetMessagesByRegion(df)
    testGetAudienceByRegion(df)

testWeek2()
do_week2()

#### FULL ASSIGNMENT ####

"""
Write a function graphAttacksAllStates(dict) which takes a dictionary containing 
the counts of how many attack tweets were sent per state, and outputs a bar chart of 
the data. Use the provided function bar_plot, and don't forget to make a good title.
"""
def graphAttacksAllStates(dict):
    return bar_plot(dict, "AttacksByState")

"""
Write a function graphTopN(dict, n, title) which takes a dictionary including keys 
from all states and an n for the number of top states to select. It should create 
a new dictionary that includes only the top n states that have the highest counts 
in the original dictionary. Then create a bar chart of those states using bar_plot.
Don't forget to include the provided title!

Hint: it may be useful to loop through the original dictionary n times, each time 
finding the top state that isn't in the new dictionary already.
"""
def graphTopN(dict, n, title):
    topStates = {}
    for i in range (0,n):
        print(topStates)
        value = 0
        state = ''
        for key in dict:
            if key not in topStates:
                if dict[key] > value:
                    state = key
                    value = dict[key]
        topStates[state] = value
    return bar_plot(topStates, title)
    
"""
Write a function graph2Regions(dict, r1, r2, title) which takes a dictionary of 
dictionaries and two regions r1 and r2 (which are keys in the dictionary), finds 
the dictionary values of those regions, and graphs those values in a side-by-side 
bar chart. In order to do this, you will need to make lists out of the dictionaries.

Make a list of all the keys in r1 and r2, with no duplicates. Then make two lists: 
one of the dictionary values in r1, one of the values in r2 (or 0 if a key doesn't 
appear). Each value at index i of the two lists should correspond to the key at 
index i of the original list.

Finally, use sidebyside_bar_plots(names, values1, values2, category1, category2, title)
with the key list as names, the two lists of values, as values1 and values2, the 
category names (r1 and r2), and the given title, to make the graph.
"""
def graph2Regions(dict, r1, r2, title):
    messageTypes = []
    typeNumbers1 = []
    typeNumbers2 = []
    #print("this is:", r1)
    #print("this is:", r2)
    for key in dict[r1]:
        messageTypes.append(key)
        typeNumbers1.append(dict[r1][key])
    for key in dict[r2]:
        if key not in messageTypes:
            messageTypes.append(key)
            typeNumbers1.append(0)
    for key in dict[r1]:
        if key not in dict[r2]:
            typeNumbers2.append(0)
        else:
            typeNumbers2.append(dict[r2][key])
    for key in dict[r2]:
        if key not in dict[r1]:
            typeNumbers2.append(dict[r2][key])
    return sidebyside_bar_plots(messageTypes, typeNumbers1, typeNumbers2, r1, r2, title)
'''
    for key in dict[r1]:
        print("This is typeNumbers1", typeNumbers1)
        typeNumbers1.append(dict[r1][key])
    for key in dict[r2]:
        print("This is typeNumbers2", typeNumbers2)
        typeNumbers2.append(dict[r2][key])
    for key in r2List:
        print("This is messageTypes", messageTypes)
        x = r2List.index(key)
        if key != messageTypes[x]:
            messageTypes[x] = key
            typeNumbers1[x] = 0
        
        messageTypes.append(key)
        typeNumbers1.append(r1[key])
        typeNumbers2.append(rs[key])
'''

"""
Write a function graphSentCountAttackCount(sentiments, attacks, title) which gets 
two dictionaries, one mapping states to negative sentiment counts, the other 
mapping states to attack counts, and creates two lists - one for the values of 
negative sentiment counts and one for values of attack counts for each 
corresponding state at the same index. Append 0 if the state is not in the 
dictionary. Then, you should use sidebyside_bar_plot to graph the corresponding 
values, similarly to how you created the graph in the previous function.
"""
def graphSentCountAttackCount(sentiments, attacks, title):
    states = []
    sentimentsCount = []
    attacksCount = []
    for state in sentiments:
        states.append(state)
        sentimentsCount.append(sentiments[state])
    for state in attacks:
        if state not in states:
            states.append(state)
            sentimentsCount.append(0)
    for state in sentiments:
        if state not in attacks:
            attacksCount.append(0)
        else:
            attacksCount.append(attacks[state])
    for state in attacks:
        if state not in sentiments:
            attacksCount.append(attacks[state])
    return sidebyside_bar_plots(states, sentimentsCount, attacksCount, "sentiment", "attack", title)
'''
    for state in states:
        print("This is sentimentsCount", sentimentsCount)
        if state not in sentiments:
           sentimentsCount.append(0) 
        else:
            sentimentsCount.append(sentiments[state])
    for state in attacks:
        print("This is attacksCount", attacksCount)
        if state not in states:
            attacksCount.append(0)
        else:
            attacksCount.append(attacks[state])  
'''  


def do_week3():
    df = makeDataFrame("politicaldata.csv")
    state_df = makeDataFrame("statemappings.csv")
    addColumns(df,state_df)
    addSentimentColumn(df)
    
    nsbs = getNegSentimentByState(df)
    abs = getAttacksByState(df)
    pbs = getPartisanByState(df)
    mbr = getMessagesByRegion(df)
    abr = getAudienceByRegion(df)
    
    graphAttacksAllStates(abs)
    graphTopN(abs, 5, "Top Attacks")
    graphTopN(pbs, 5, "Top Partisan Messages")
    graph2Regions(mbr, "West", "South", "Messages by Region")
    graph2Regions(abr, "West", "South", "Audience by Region")
    graphSentCountAttackCount(nsbs, abs, "Sentiment vs Attacks by State")
    
#### WEEK 3 PROVIDED CODE ####

"""
Expects a dictionary of states as keys with counts as values, and a title.
Plots the states on the x axis, counts as the y axis and puts a title on top.
"""
def bar_plot(dict, title):
    names = list(dict.keys())
    values = list(dict.values())
    plt.bar(names, values)
    plt.xticks(names, rotation='vertical')
    plt.title(title)
    plt.show()

"""
Expects 3 lists - one of names, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of counts side by side to look at the differences.
"""
def sidebyside_bar_plots(names, values1, values2, category1, category2, title):
    x = list(range(len(names)))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    pos1 = []
    pos2 = []
    for i in x:
        pos1.append(i - width/2)
        pos2.append(i + width/2)
    rects1 = ax.bar(pos1, values1, width, label=category1)
    rects2 = ax.bar(pos2, values2, width, label=category2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    plt.title(title)
    plt.xticks(rotation="vertical")
    fig.tight_layout()
    plt.show()

#### WEEK 3 TESTS ####

# Instead of running individual tests, check the new graph generated by do_week3
# after you finish each function.

do_week3()
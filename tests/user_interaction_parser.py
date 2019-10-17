import json

from user_interaction.user_interaction_parser import parse_users_interactions


def test_parse_users_interactions():
    # sanity
    tree = {"node": {
        "text": "[deleted]",
        "id": "7f53sk",
        "author": "[deleted]",
        "extra_data": {
            "subreddit": "changemyview",
            "subreddit_id": "t5_2w2s8",
            "permalink": "/r/changemyview/comments/7f53sk/cmv_the_biggest_issue_in_america_is_the/",
            "url": "https://www.reddit.com/r/changemyview/comments/7f53sk/cmv_the_biggest_issue_in_america_is_the/",
            "title": "CMV: The biggest issue in America is the economical system and general wage system.",
            "num_comments": 5
        },
        "timestamp": 1511499160
    },
        "children": [
            {
                "node": {
                    "text": "I'm not going to argue that the economics and the state of income inequality isn't a huge deal, but there are definitely more factors than just the minimum wage. Your California example is due to a ton of different factors, including massively distorted localized housing, California's absurd system for (not) developing new houses, and California's equally absurd property tax system. So while I agree that far fewer people can afford their own homes, I think it's a bit weird to specifically pick a hyperinflated market for rich people and use that as an example for the general public. In the general public market, property prices *are* relatively higher and there are fewer small \"starter homes\" because instead you can sell 3+ bedroom 2+ bath houses, which is an issue in terms of letting people buy property to build any sort of equity. But the American ideal of owning a house as proof of success is also kinda weird and I think it's better to look at things more fundamentally than \"can you afford property\"; being able to afford basic necessities, some quality of life goods, and healthcare seem more important than whether you own a 3-bedroom house for yourself or rent somewhere. Additionally, the \"forget social/political issues and forget diplomacy\" are a little odd, because those directly impact economics. Social issues have a huge impact on a number of people, including how those people are treated economically. A rising tide may lift all boats, but if some people don't have a boat (why did I use a metaphor, I hate metaphors) due to social factors like massively higher arrest rates or legalized employment discrimination, they don't see the benefit. Being able to say \"ignore social issues, just focus on economic ones\" is only helpful if social issues are not affecting you directly. Likewise, foreign policy is... pretty important for economics? I mean it's also super arcane and complex so I don't blame anybody for not knowing about it, and I certainly don't know enough about it to speak authoritatively. But the very fact an American lifestyle with cheap clothing, luxury goods, etc. is built on the backs of trade agreements and third-world labor/manufacturing means that exactly how those things are set up is pretty important.",
                    "id": "dq9kszi",
                    "author": "Milskidasith",
                    "extra_data": {
                        "subreddit_id": "t5_2w2s8",
                        "subreddit": "changemyview",
                        "parent_id": "t3_7f53sk",
                        "link_id": "t3_7f53sk",
                        "permalink": "/r/changemyview/comments/7f53sk/cmv_the_biggest_issue_in_america_is_the/dq9kszi/"
                    },
                    "timestamp": 1511500750
                },
                "children": []
            },
            {
                "node": {
                    "text": "I'd argue that our political problems outweigh the importance of our economics. This is simply because it is our politicians who *control* our economics, and politics is permeated by issues like: * alarmism * corporate interest * dishonesty * self-interest * the degradation of our checks and balances system * foreign influence that ultimately does not allow for economic policies to be passed, on any level, that are truly in the best interest of the populace. This is either through outright blocking these policies or preventing the election of people who will pass them. Let's assume raising the minimum wage is the policy at hand, and that we believe it will be in the best interest of the general populace. * Alarmists will promote this as a sign of socialism (often used as a buzzword in American politics) * It is not in the best interest of corporations for this bill to be passed * It will not be in the interest of right-wing politicians to vote for this bill, as they may lose support from both their donors and their party. See how the politics ultimately controls the economy?",
                    "id": "dq9m8km",
                    "author": "DangerousHarvey",
                    "extra_data": {
                        "subreddit_id": "t5_2w2s8",
                        "subreddit": "changemyview",
                        "parent_id": "t3_7f53sk",
                        "link_id": "t3_7f53sk",
                        "permalink": "/r/changemyview/comments/7f53sk/cmv_the_biggest_issue_in_america_is_the/dq9m8km/"
                    },
                    "timestamp": 1511503414
                },
                "children": [
                    {
                        "node": {
                            "text": "So you're saying that our foreign occupation and commodations for sexual preferences echo significantly to our overall economy and has superior direct effect on our wage system?",
                            "id": "dq9mxr5",
                            "author": "mergerr",
                            "extra_data": {
                                "subreddit_id": "t5_2w2s8",
                                "subreddit": "changemyview",
                                "parent_id": "t1_dq9m8km",
                                "link_id": "t3_7f53sk",
                                "permalink": "/r/changemyview/comments/7f53sk/cmv_the_biggest_issue_in_america_is_the/dq9mxr5/"
                            },
                            "timestamp": 1511504807
                        },
                        "children": [
                            {
                                "node": {
                                    "text": "No. I wasn't talking about other policies that may impact our economy. More what I meant was \"The issues we have with *the way our politics are conducted* (see my above comment) makes it nigh on impossible to effectively solve our economic problems. Therefore, the former is more important than the latter.\"",
                                    "id": "dq9n35j",
                                    "author": "DangerousHarvey",
                                    "extra_data": {
                                        "subreddit_id": "t5_2w2s8",
                                        "subreddit": "changemyview",
                                        "parent_id": "t1_dq9mxr5",
                                        "link_id": "t3_7f53sk",
                                        "permalink": "/r/changemyview/comments/7f53sk/cmv_the_biggest_issue_in_america_is_the/dq9n35j/"
                                    },
                                    "timestamp": 1511505113
                                },
                                "children": []
                            }
                        ]
                    }
                ]
            },
            {
                "node": {
                    "text": "You haven't done much to actually argue in favor of your view. Why do you think those three issues are more important than every other?",
                    "id": "dq9nfsx",
                    "author": "PreacherJudge",
                    "extra_data": {
                        "subreddit_id": "t5_2w2s8",
                        "subreddit": "changemyview",
                        "parent_id": "t3_7f53sk",
                        "link_id": "t3_7f53sk",
                        "permalink": "/r/changemyview/comments/7f53sk/cmv_the_biggest_issue_in_america_is_the/dq9nfsx/"
                    },
                    "timestamp": 1511505843
                },
                "children": [
                    {
                        "node": {
                            "text": "[deleted]",
                            "id": "dq9nuhl",
                            "author": "[deleted]",
                            "extra_data": {
                                "subreddit_id": "t5_2w2s8",
                                "subreddit": "changemyview",
                                "parent_id": "t1_dq9nfsx",
                                "link_id": "t3_7f53sk",
                                "permalink": "/r/changemyview/comments/7f53sk/cmv_the_biggest_issue_in_america_is_the/dq9nuhl/"
                            },
                            "timestamp": 1511506704
                        },
                        "children": []
                    }
                ]
            }
        ]
    }

    interactions = parse_users_interactions(tree)
    print(json.dumps(interactions, indent=4, default=lambda cls: cls.__dict__))


if __name__ == "__main__":
    test_parse_users_interactions()


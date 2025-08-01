from nsql.database import NeuralDB
g_data_item={}
g_data_item['table']={
        "id": 12322,
        "header": [
          "Year",
          "Single",
          "US Country",
          "CAN Country",
          "Album"
        ],
        "rows": [
          [
            "1985",
            "\"Playing for Keeps\"",
            "62",
            "\u2014",
            "N/A"
          ],
          [
            "1985",
            "\"My Heart Holds On\"",
            "64",
            "\u2014",
            "Holly Dunn"
          ],
          [
            "1986",
            "\"Two Too Many\"",
            "39",
            "\u2014",
            "Holly Dunn"
          ],
          [
            "1986",
            "\"Daddy's Hands\"",
            "7",
            "\u2014",
            "Holly Dunn"
          ],
          [
            "1987",
            "\"Love Someone Like Me\"",
            "2",
            "2",
            "Cornerstone"
          ],
          [
            "1987",
            "\"Only When I Love\"",
            "4",
            "7",
            "Cornerstone"
          ],
          [
            "1988",
            "\"Strangers Again\"",
            "7",
            "36",
            "Cornerstone"
          ],
          [
            "1988",
            "\"That's What Your Love Does to Me\"",
            "5",
            "6",
            "Across the Rio Grande"
          ],
          [
            "1988",
            "\"(It's Always Gonna Be) Someday\"",
            "11",
            "N/A",
            "Across the Rio Grande"
          ],
          [
            "1989",
            "\"Are You Ever Gonna Love Me\"",
            "1",
            "6",
            "The Blue Rose of Texas"
          ],
          [
            "1989",
            "\"There Goes My Heart Again\"",
            "4",
            "8",
            "The Blue Rose of Texas"
          ],
          [
            "1990",
            "\"My Anniversary for Being a Fool\"",
            "63",
            "75",
            "Heart Full of Love"
          ],
          [
            "1990",
            "\"You Really Had Me Going\"",
            "1",
            "1",
            "Heart Full of Love"
          ],
          [
            "1991",
            "\"Heart Full of Love\"",
            "19",
            "12",
            "Heart Full of Love"
          ],
          [
            "1991",
            "\"Maybe I Mean Yes\"",
            "48",
            "45",
            "Milestones: Greatest Hits"
          ],
          [
            "1991",
            "\"No One Takes the Train Anymore\"",
            "\u2014",
            "\u2014",
            "Milestones: Greatest Hits"
          ],
          [
            "1992",
            "\"No Love Have I\"",
            "67",
            "\u2014",
            "Getting It Dunn"
          ],
          [
            "1992",
            "\"As Long as You Belong to Me\"",
            "68",
            "\u2014",
            "Getting It Dunn"
          ],
          [
            "1992",
            "\"Golden Years\"",
            "51",
            "62",
            "Getting It Dunn"
          ],
          [
            "1995",
            "\"I Am Who I Am\"",
            "56",
            "56",
            "Life and Love and All the Stages"
          ],
          [
            "1995",
            "\"Cowboys Are My Weakness\"",
            "\u2014",
            "\u2014",
            "Life and Love and All the Stages"
          ],
          [
            "1995",
            "\"It's Not About Blame\"",
            "\u2014",
            "\u2014",
            "Life and Love and All the Stages"
          ],
          [
            "1997",
            "\"Leave One Bridge Standing\"",
            "\u2014",
            "\u2014",
            "Leave One Bridge Standing"
          ],
          [
            "\"\u2014\" denotes releases that did not chart",
            "\"\u2014\" denotes releases that did not chart",
            "\"\u2014\" denotes releases that did not chart",
            "\"\u2014\" denotes releases that did not chart",
            "\"\u2014\" denotes releases that did not chart"
          ]
        ],
        "page_title": "Holly Dunn"
}

g_data_item['table']
db = NeuralDB(
                tables=[{'title': '222', 'table': g_data_item['table']}]
            )
g_data_item['table'] = db.get_table_df()#<class 'pandas.core.frame.DataFrame'>
g_data_item['title'] = db.get_table_title() #<class 'str'>


print(f"table:{g_data_item['table']} \n  title :{g_data_item['title']} ")

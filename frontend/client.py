from docarray import Document, DocumentArray
from jina import Client
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
def get_matches_from_image(input:str):
    # data = input.read()
    # query_doc = Document(blob=data)

    client = Client(host="0.0.0.0:49631")
    response = client.search(
        input,
        return_results=True,
        parameters={"limit": 0, "filter": None},
        show_progress=True,
    )
    for match in response[0].matches:
        print(f'({match.scores["cosine"].value}) id: {match.id} tags: {dict(match.tags)}')
        img = Image.open(match.uri)
        imshow(img)
        plt.title(f'{match.tags["productDisplayName"]}')
        plt.show()
get_matches_from_image("dress")
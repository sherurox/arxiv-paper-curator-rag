import uuid
import sys
sys.path.insert(0, '/app')

from src.services.opensearch.factory import make_opensearch_client
from src.db.factory import make_database
from src.repositories.paper import PaperRepository

db = make_database()
client = make_opensearch_client()

dummy_embedding = [0.001] * 1024

with db.get_session() as session:
    repo = PaperRepository(session)
    papers = repo.get_all()
    print(f'Found {len(papers)} papers in DB')
    indexed = 0
    for paper in papers:
        pub_date = None
        if paper.published_date:
            pub_date = paper.published_date.strftime('%Y-%m-%dT%H:%M:%S')
        chunk = {
            'arxiv_id': paper.arxiv_id,
            'title': paper.title,
            'abstract': paper.abstract or '',
            'chunk_text': str(paper.title) + '. ' + str(paper.abstract or ''),
            'chunk_id': str(uuid.uuid4()),
            'authors': paper.authors or [],
            'categories': paper.categories or [],
            'published_date': pub_date,
            'section_title': 'abstract',
            'embedding': dummy_embedding,
        }
        try:
            response = client.client.index(index=client.index_name, body=chunk, refresh=True)
            if response['result'] in ['created', 'updated']:
                indexed += 1
        except Exception as e:
            print(f'Error: {e}')
    print(f'Indexed {indexed} of {len(papers)} papers successfully')

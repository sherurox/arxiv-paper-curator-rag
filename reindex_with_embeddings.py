import uuid
import sys
import asyncio
sys.path.insert(0, '/app')

from src.services.opensearch.factory import make_opensearch_client
from src.services.embeddings.factory import make_embeddings_service
from src.db.factory import make_database
from src.repositories.paper import PaperRepository

async def main():
    db = make_database()
    opensearch_client = make_opensearch_client()
    embeddings_service = make_embeddings_service()

    with db.get_session() as session:
        repo = PaperRepository(session)
        papers = repo.get_all()
        print(f'Found {len(papers)} papers in DB')

        # Delete existing index and recreate
        try:
            opensearch_client.client.indices.delete(index=opensearch_client.index_name)
            print('Deleted existing index')
        except Exception:
            pass

        setup = opensearch_client.setup_indices(force=True)
        print(f'Index recreated: {setup}')

        indexed = 0
        for i, paper in enumerate(papers):
            chunk_text = str(paper.title) + '. ' + str(paper.abstract or '')

            # Get real embedding from Jina
            try:
                embeddings = await embeddings_service.embed_passages([chunk_text])
                embedding = embeddings[0] if embeddings else [0.001] * 1024
            except Exception as e:
                print(f'Embedding error for {paper.arxiv_id}: {e}')
                embedding = [0.001] * 1024

            pub_date = None
            if paper.published_date:
                pub_date = paper.published_date.strftime('%Y-%m-%dT%H:%M:%S')

            chunk = {
                'arxiv_id': paper.arxiv_id,
                'title': paper.title,
                'abstract': paper.abstract or '',
                'chunk_text': chunk_text,
                'chunk_id': str(uuid.uuid4()),
                'authors': paper.authors or [],
                'categories': paper.categories or [],
                'published_date': pub_date,
                'section_title': 'abstract',
                'embedding': embedding,
            }

            try:
                response = opensearch_client.client.index(
                    index=opensearch_client.index_name,
                    body=chunk,
                    refresh=True
                )
                if response['result'] in ['created', 'updated']:
                    indexed += 1
                    print(f'[{i+1}/{len(papers)}] Indexed: {paper.arxiv_id}')
            except Exception as e:
                print(f'Index error for {paper.arxiv_id}: {e}')

        print(f'Done. Indexed {indexed} of {len(papers)} papers with real embeddings')

asyncio.run(main())

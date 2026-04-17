import logging
from typing import List, Dict, Any, Optional
from services.client import supabase

logger = logging.getLogger(__name__)

class DBService:
    @staticmethod
    def create_document(user_id: str, filename: str, storage_path: str) -> str:
        """Creates a document record and returns the ID."""
        try:
            data = {
                "user_id": user_id,
                "filename": filename,
                "storage_path": storage_path,
                "processed": False
            }
            response = supabase.table("documents").insert(data).execute()
            return response.data[0]['id']
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            raise

    @staticmethod
    def update_document_processed(document_id: str, status: bool = True):
        try:
            supabase.table("documents").update({"processed": status}).eq("id", document_id).execute()
        except Exception as e:
            logger.error(f"Error updating document status: {e}")
            raise

    @staticmethod
    def add_chunks(chunks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Bulk insert chunks.
        chunks_data should list of dicts with:
        - document_id
        - chunk_index
        - text_content
        - summary (optional)
        """
        try:
            response = supabase.table("chunks").insert(chunks_data).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            raise

    @staticmethod
    def add_topics(topics: List[str]) -> Dict[str, str]:
        """
        Ensures topics exist and returns a map of {name: id}.
        """
        topic_map = {}
        if not topics:
            return topic_map

        # 1. Fetch existing
        try:
            response = supabase.table("topics").select("id, name").in_("name", topics).execute()
            existing = {row['name']: row['id'] for row in response.data}
            topic_map.update(existing)

            # 2. Identify missing
            missing = [t for t in topics if t not in existing]
            
            # 3. Insert missing
            if missing:
                new_topic_rows = [{"name": t} for t in missing]
                # Insert and return inserted
                insert_resp = supabase.table("topics").insert(new_topic_rows).execute()
                for row in insert_resp.data:
                    topic_map[row['name']] = row['id']
            
            return topic_map
        except Exception as e:
            logger.error(f"Error managing topics: {e}")
            raise

    @staticmethod
    def link_chunk_topics(links: List[Dict[str, str]]):
        """
        Link chunks to topics.
        links: list of {'chunk_id': uuid, 'topic_id': uuid}
        """
        if not links:
            return
        try:
            supabase.table("chunk_topics").insert(links).execute()
        except Exception as e:
            logger.error(f"Error linking chunk topics: {e}")
            # Identify which one failed? For now just raise.
            raise

    @staticmethod
    def add_faiss_map(faiss_index: int, chunk_id: str):
        try:
            supabase.table("faiss_map").insert({
                "faiss_index": faiss_index,
                "chunk_id": chunk_id
            }).execute()
        except Exception as e:
            logger.error(f"Error adding faiss map: {e}")
            raise
    
    @staticmethod
    def get_chunk_by_faiss_index(faiss_index: int) -> Optional[Dict]:
        try:
            # Join faiss_map -> chunks
            # Supabase join syntax: select(..., count:exact)
            resp = supabase.table("faiss_map").select("chunk_id, chunks(*)").eq("faiss_index", faiss_index).execute()
            if resp.data:
                # Structure: [{'chunk_id': '...', 'chunks': {'id': '...', 'text_content': '...'}}]
                return resp.data[0].get('chunks')
            return None
        except Exception as e:
            logger.error(f"Error fetching chunk by faiss index: {e}")
            return None

    @staticmethod
    def get_topics_by_filename(filename: str) -> List[str]:
        try:
            # 1. Get Doc ID (Most recent first)
            doc_resp = supabase.table("documents").select("id").eq("filename", filename).order("created_at", desc=True).execute()
            if not doc_resp.data:
                return []
            
            # Use most recent
            doc_id = doc_resp.data[0]['id']

            # 2. Get Topics via Chunks
            chunks_resp = supabase.table("chunks").select("id").eq("document_id", doc_id).execute()
            chunk_ids = [c['id'] for c in chunks_resp.data]
            
            if not chunk_ids:
                return []
                
            # Get topic IDs linked to these chunks
            ct_resp = supabase.table("chunk_topics").select("topic_id").in_("chunk_id", chunk_ids).execute()
            topic_ids = list(set([row['topic_id'] for row in ct_resp.data]))
            
            if not topic_ids:
                return []
                
            # Get Topic Names
            t_resp = supabase.table("topics").select("name").in_("id", topic_ids).execute()
            return [row['name'] for row in t_resp.data]
            
        except Exception as e:
            logger.error(f"Error fetching topics for file: {e}")
            return []

    @staticmethod
    def get_chunks_by_topic(topic_name: str) -> List[Dict[str, Any]]:
        try:
            # 1. Get Topic ID
            t_res = supabase.table("topics").select("id").eq("name", topic_name).execute()
            if not t_res.data:
                return []
            topic_id = t_res.data[0]['id']
            
            # 2. Get Chunk IDs
            ct_res = supabase.table("chunk_topics").select("chunk_id").eq("topic_id", topic_id).execute()
            chunk_ids = [row['chunk_id'] for row in ct_res.data]
            
            if not chunk_ids:
                return []
                
            # 3. Get Chunks
            # Supabase 'in' filter
            c_res = supabase.table("chunks").select("*").in_("id", chunk_ids).execute()
            return c_res.data
            
        except Exception as e:
            logger.error(f"Error fetching chunks by topic: {e}")
            return []

    @staticmethod
    def get_documents_by_user(user_id: str) -> List[Dict[str, Any]]:
        try:
            # 1. Get Admin IDs
            admin_res = supabase.table("users").select("id").eq("role", "admin").execute()
            admin_ids = [row['id'] for row in admin_res.data]
            
            # 2. Build list of allowed IDs (User + Admins)
            allowed_ids = list(set([user_id] + admin_ids))
            
            # 3. Fetch documents owned by these users
            response = supabase.table("documents").select("*").in_("user_id", allowed_ids).order("created_at", desc=True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching user documents: {e}")
            return []

    @staticmethod
    def get_all_documents() -> List[Dict[str, Any]]:
        try:
            # For admin panel: list all unique filenames or all docs
            response = supabase.table("documents").select("*").order("created_at", desc=True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching all documents: {e}")
            return []

    @staticmethod
    def delete_document(document_id: str):
        try:
            # 1. Get storage path
            res = supabase.table("documents").select("storage_path").eq("id", document_id).execute()
            if res.data:
                path = res.data[0]['storage_path']
                # Delete from Storage
                try:
                    supabase.storage.from_("pdfs").remove(path)
                except:
                    logger.warning(f"Could not delete {path} from storage")
            
            # 2. Delete from DB (Cascade should handle chunks/topics)
            supabase.table("documents").delete().eq("id", document_id).execute()
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise

    @staticmethod
    def add_kg_node(entity_name: str, document_id: str, chunk_id: str):
        try:
            res = supabase.table("knowledge_graph_nodes").select("id, chunk_ids").eq("entity_name", entity_name).execute()
            if res.data:
                node = res.data[0]
                chunk_ids = node.get("chunk_ids", [])
                if chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)
                    supabase.table("knowledge_graph_nodes").update({"chunk_ids": chunk_ids}).eq("id", node["id"]).execute()
                return node["id"]
            else:
                res = supabase.table("knowledge_graph_nodes").insert({
                    "entity_name": entity_name,
                    "document_id": document_id,
                    "chunk_ids": [chunk_id]
                }).execute()
                return res.data[0]["id"]
        except Exception as e:
            logger.error(f"Error adding KG node: {e}")
            return None

    @staticmethod
    def add_kg_edge(source_entity: str, target_entity: str, relation_label: str, weight: float = 1.0):
        try:
            res_src = supabase.table("knowledge_graph_nodes").select("id").eq("entity_name", source_entity).execute()
            res_tgt = supabase.table("knowledge_graph_nodes").select("id").eq("entity_name", target_entity).execute()
            
            if not res_src.data or not res_tgt.data:
                return None
                
            src_id = res_src.data[0]["id"]
            tgt_id = res_tgt.data[0]["id"]
            
            res = supabase.table("knowledge_graph_edges").select("id, weight").eq("source_node_id", src_id).eq("target_node_id", tgt_id).eq("relation_label", relation_label).execute()
            
            if res.data:
                edge = res.data[0]
                new_weight = edge["weight"] + weight
                supabase.table("knowledge_graph_edges").update({"weight": new_weight}).eq("id", edge["id"]).execute()
            else:
                supabase.table("knowledge_graph_edges").insert({
                    "source_node_id": src_id,
                    "target_node_id": tgt_id,
                    "relation_label": relation_label,
                    "weight": weight
                }).execute()
        except Exception as e:
            logger.error(f"Error adding KG edge: {e}")

    @staticmethod
    def get_kg_related_chunks(concepts: List[str]) -> List[Dict[str, Any]]:
        try:
            if not concepts:
                return []
            res = supabase.table("knowledge_graph_nodes").select("id, chunk_ids").in_("entity_name", concepts).execute()
            if not res.data:
                return []
            chunk_ids = set()
            for node in res.data:
                for cid in node.get("chunk_ids", []):
                    chunk_ids.add(cid)
            if not chunk_ids:
                return []
            c_res = supabase.table("chunks").select("*").in_("id", list(chunk_ids)).execute()
            return c_res.data
        except Exception as e:
            logger.error(f"Error fetching KG related chunks: {e}")
            return []

import io
import os
import unittest
from app import app

class TestChunkedUpload(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_chunked_upload_and_complete(self):
        upload_id = "test_upload_12345"
        filename = "test_image.png"
        
        from PIL import Image
        buf = io.BytesIO()
        Image.new('RGB', (200, 200), color='blue').save(buf, format='PNG')
        png_bytes = buf.getvalue()
        
        chunk_size = 500
        total_chunks = (len(png_bytes) + chunk_size - 1) // chunk_size
        
        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(png_bytes))
            chunk_data = png_bytes[start:end]
            
            resp = self.client.post('/api/agent/upload/chunk', data={
                'file': (io.BytesIO(chunk_data), filename),
                'upload_id': upload_id,
                'chunk_index': str(i),
                'total_chunks': str(total_chunks),
                'filename': filename
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertEqual(data.get('status'), 'chunk_received')
            self.assertEqual(data.get('chunk_index'), i)

        # Complete the upload
        complete_resp = self.client.post('/api/agent/upload/complete', json={
            'upload_id': upload_id,
            'filename': filename,
            'total_chunks': total_chunks
        })
        self.assertEqual(complete_resp.status_code, 200)
        final_data = complete_resp.get_json()
        self.assertEqual(final_data.get('status'), 'success')
        self.assertEqual(final_data.get('type'), 'image')
        self.assertTrue(final_data.get('url').startswith('/media/agent_'))
        
        # Check that saved file exists
        saved_file = final_data.get('filename')
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_file)
        self.assertTrue(os.path.exists(saved_path))
        self.assertEqual(os.path.getsize(saved_path), len(png_bytes))
        
        # Clean up
        if os.path.exists(saved_path):
            os.remove(saved_path)

    def test_chunked_upload_abort(self):
        upload_id = "test_upload_abort_999"
        filename = "test_image.png"

        resp = self.client.post('/api/agent/upload/chunk', data={
            'file': (io.BytesIO(b"part0"), filename),
            'upload_id': upload_id,
            'chunk_index': '0',
            'total_chunks': '2',
            'filename': filename
        })
        self.assertEqual(resp.status_code, 200)

        chunk_dir = os.path.join(app.config['UPLOAD_FOLDER'], '.chunks', upload_id)
        self.assertTrue(os.path.isdir(chunk_dir))

        abort_resp = self.client.post('/api/agent/upload/abort', json={'upload_id': upload_id})
        self.assertEqual(abort_resp.status_code, 200)
        self.assertFalse(os.path.isdir(chunk_dir))

    def test_chunked_upload_missing_parts(self):
        upload_id = "test_upload_incomplete_888"
        filename = "test_image.png"

        # Upload part 0 of 2, but skip part 1
        resp = self.client.post('/api/agent/upload/chunk', data={
            'file': (io.BytesIO(b"part0"), filename),
            'upload_id': upload_id,
            'chunk_index': '0',
            'total_chunks': '2',
            'filename': filename
        })
        self.assertEqual(resp.status_code, 200)

        # Attempt to complete prematurely
        complete_resp = self.client.post('/api/agent/upload/complete', json={
            'upload_id': upload_id,
            'filename': filename,
            'total_chunks': 2
        })
        self.assertEqual(complete_resp.status_code, 400)
        err = complete_resp.get_json().get('error')
        self.assertIn("Missing chunk part 1 of 2", err)

        # Abort and clean up
        self.client.post('/api/agent/upload/abort', json={'upload_id': upload_id})

if __name__ == '__main__':
    unittest.main()

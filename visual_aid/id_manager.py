class Tracker_Id_Manager:
    def __init__(self):
        self.id_mapping = {}
        self.active_ids= set()
    
    def update_mapping(self,current_ids):
        self.active_ids = set(current_ids)
    
    def get_active_ids(self):
        return list(self.active_ids)
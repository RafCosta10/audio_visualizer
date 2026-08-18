use mpris::PlayerFinder;
use std::time::Duration;

fn main() {
    let finder = PlayerFinder::new().unwrap();
    if let Ok(mut players) = finder.find_all() {
        if let Some(player) = players.into_iter().next() {
            if let Ok(metadata) = player.get_metadata() {
                if let Some(track_id) = metadata.track_id() {
                    let _ = player.set_position(track_id, &Duration::from_secs(10));
                }
            }
        }
    }
}

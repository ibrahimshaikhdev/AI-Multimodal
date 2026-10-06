from datetime import datetime, timezone

from backend.extensions import db


class ResearchPaper(db.Model):
    __tablename__ = "research_papers"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = db.Column(db.String(200), nullable=False)
    topic = db.Column(db.String(120), nullable=False)
    paper_date = db.Column(db.Date, nullable=True)
    authors = db.Column(db.String(255), nullable=True)
    journal = db.Column(db.String(255), nullable=True)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(80), nullable=False, unique=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self):
        return f"<ResearchPaper {self.id}: {self.title}>"


__all__ = ["ResearchPaper"]

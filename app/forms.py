from flask.ext.wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired


class SettingsForm(Form):
    filename = StringField('filename', validators=[DataRequired()])
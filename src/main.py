from telegram.ext import Updater
from telegram.ext import CommandHandler, ConversationHandler, MessageHandler, Filters
from telegram import ParseMode
import logging
import sys
sys.path.append("..")
from nlp_model.model import model

updater = Updater(token='', use_context=True)
dispatcher = updater.dispatcher

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

MSG = 0

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=str(update.message.from_user))

def help(update, context):
    update.message.reply_text('/become\_member use this to register yourself\n'
    '/submit\_problem use this to submit a problem or an update to a problem\n'
    '/cancel use this to stop the operation for problem submission', parse_mode=ParseMode.MARKDOWN)

def become_member(update, context):
    update.message.reply_text('In order to become a StuStaNet member\n'
    'visit the following website:\n'
    'reg.stustanet.de')

def problem(update, context):

    update.message.reply_text('Now, tell me your problem.')
    return MSG

def process(update, context):
    classifier, vec = model("../data/data.csv")
    processed_msg = vec.transform([update.message.text]).toarray()
    prediction = classifier.predict(processed_msg)
    if prediction=='positive':
        update.message.reply_text('Glad that your problem has been solved!')
    else:
        update.message.reply_text('Ok, we will look into this problem.')

def cancel(update, context):
    update.message.reply_text('Issue posting cancelled.')

    return ConversationHandler.END

def main():

    start_handler = CommandHandler('start', start, run_async=True)
    dispatcher.add_handler(start_handler)

    help_handler = CommandHandler('help', help, run_async=True)
    dispatcher.add_handler(help_handler)

    member_handler = CommandHandler('become_member', become_member, run_async=True)
    dispatcher.add_handler(member_handler)

    problem_handler = ConversationHandler(entry_points=[CommandHandler('submit_problem', problem, run_async=True)],
    states={MSG: [MessageHandler(Filters.text & ~Filters.command, process, run_async=True)]},
    fallbacks=[CommandHandler('cancel', cancel, run_async=True)])
    dispatcher.add_handler(problem_handler)
    
    updater.start_polling()

if __name__ == "__main__":
    main()
